/* Copyright 2023 The DeepRec Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=======================================================================*/

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif

#include "dynamic_embedding_server/include/ops/redistribution_functor.h"
#include "dynamic_embedding_server/include/utils/naming.h"
#include "tensorflow/core/framework/embedding/embedding_var.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

namespace des {

namespace {
const size_t kBufferSize = 8 << 20;
}

template <typename TKey, typename TValue>
class KvResourceFilterOp : public OpKernel {
 public:
  explicit KvResourceFilterOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("partition_id", &partition_id_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("partition_nums", &partition_nums_));
  }

  void Compute(OpKernelContext* ctx) override {
    EmbeddingVar<TKey, TValue>* embedding_var = nullptr;
    OP_REQUIRES_OK(
        ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &embedding_var));
    core::ScopedUnref unref_me(embedding_var);
    const Tensor& partition_num_tensor = ctx->input(1);
    int partition_num = partition_num_tensor.flat<int>()(0);

    std::vector<std::vector<TKey>> filtered_keys;
    std::vector<std::vector<void*>> value_ptr_list;
    filtered_keys.resize(partition_nums_);
    value_ptr_list.resize(partition_nums_);
    int64 before_size = embedding_var->Size();
    OP_REQUIRES_OK(
        ctx, embedding_var->GetShardedSnapshot(filtered_keys, value_ptr_list,
                                               partition_id_, partition_num));
    for (int i = 0; i < partition_nums_; ++i) {
      Tensor* unneeded_ids_tensor = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i, {filtered_keys[i].size()},
                                               &unneeded_ids_tensor));
      Tensor* unneeded_value_tensor = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(partition_nums_ + i,
                                               {filtered_keys[i].size(),
                                                embedding_var->ValueLen()},
                                               &unneeded_value_tensor));
      Tensor* version_tensor = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(2 * partition_nums_ + i,
                                               {filtered_keys[i].size()},
                                               &version_tensor));
      Tensor* freq_tensor = nullptr;
      OP_REQUIRES_OK(
          ctx, ctx->allocate_output(3 * partition_nums_ + i,
                                    {filtered_keys[i].size()}, &freq_tensor));
      if (filtered_keys[i].size() == 0) {
        continue;
      }
      auto unneeded_ids = unneeded_ids_tensor->flat<TKey>().data();
      auto unneeded_value = unneeded_value_tensor->flat<TValue>().data();
      auto versions = version_tensor->flat<int64>().data();
      auto freq = freq_tensor->flat<int64>().data();
      embedding_var->ExportAndRemove(unneeded_ids, unneeded_value, versions,
                                     freq, filtered_keys[i], value_ptr_list[i]);
    }
    int64 after_size = embedding_var->Size();
    VLOG(1) << " KvResourceFilterOp: " << embedding_var->Name()
            << " before_size " << before_size << " after_size " << after_size;
  }

 private:
  int partition_id_;
  int partition_nums_;
};

#define REGISTER_CPU_KERNELS(key_type, value_type)                  \
  REGISTER_KERNEL_BUILDER(Name(::des::kEvExportOp)                  \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<key_type>("Tkeys")    \
                              .TypeConstraint<value_type>("dtype"), \
                          KvResourceFilterOp<key_type, value_type>)

REGISTER_CPU_KERNELS(int32, float);
REGISTER_CPU_KERNELS(int64, float);
#undef REGISTER_CPU_KERNELS

template <typename TKey, typename TValue>
class KvResourceMulImportOp : public OpKernel {
 public:
  explicit KvResourceMulImportOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("partition_id", &partition_id_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("partition_nums", &partition_nums_));
  }

  void Compute(OpKernelContext* ctx) override {
    EmbeddingVar<TKey, TValue>* embedding_var = nullptr;
    OP_REQUIRES_OK(
        ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &embedding_var));
    core::ScopedUnref unref_me(embedding_var);
    const Tensor& partition_num_tensor = ctx->input(1);
    int partition_num = partition_num_tensor.flat<int>()(0);
    int64 before_size = embedding_var->Size();
    if (partition_id_ < partition_num) {
      // part 0 is itself, skipped.
      for (int i = 1; i < partition_nums_; ++i) {
        const Tensor& import_ids_tensor = ctx->input(2 + i);
        auto* import_ids = import_ids_tensor.flat<TKey>().data();
        int64 N = import_ids_tensor.NumElements();
        if (N == 0) continue;
        const Tensor& import_values_tensor =
            ctx->input(2 + partition_nums_ + i);
        auto* import_values = import_values_tensor.flat<float>().data();

        const Tensor& import_versions_tensor =
            ctx->input(2 + partition_nums_ * 2 + i);
        auto* import_versions = import_versions_tensor.flat<int64>().data();
        const Tensor& import_freqs_tensor =
            ctx->input(2 + partition_nums_ * 3 + i);
        auto* import_freqs = import_freqs_tensor.flat<int64>().data();
        OP_REQUIRES_OK(ctx, embedding_var->RestoreFromKeysAndValues(
                                N, partition_id_, partition_num, import_ids,
                                import_values, import_versions, import_freqs));

        while (N > 0) {
          int64 read_num = std::min(N, (int64)kBufferSize);
          OP_REQUIRES_OK(ctx,
                         embedding_var->RestoreFromKeysAndValues(
                             read_num, partition_id_, partition_num, import_ids,
                             import_values, import_versions, import_freqs));
          N -= read_num;
        }
      }
      int64 after_size = embedding_var->Size();
      VLOG(1) << " KvResourceMulImport: " << embedding_var->Name()
              << " after_size " << after_size << " before_size " << before_size;
    }
  }

 private:
  int partition_id_;
  int partition_nums_;
};

#define REGISTER_CPU_KERNELS(key_type, value_type)                  \
  REGISTER_KERNEL_BUILDER(Name(::des::kEvImportOp)                  \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<key_type>("Tkeys")    \
                              .TypeConstraint<value_type>("dtype"), \
                          KvResourceMulImportOp<key_type, value_type>)

REGISTER_CPU_KERNELS(int32, float);
REGISTER_CPU_KERNELS(int64, float);
#undef REGISTER_CPU_KERNELS

template <typename Device, typename T>
class ReAssignOp : public OpKernel {
 public:
  explicit ReAssignOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(context, context->GetAttr("partition_id", &partition_id_));
    OP_REQUIRES_OK(context, context->GetAttr("device_id", &device_id_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("partition_nums", &num_partitions_));
    OP_REQUIRES(context, IsRefType(context->input_type(0)),
                errors::InvalidArgument("lhs input needs to be a ref type"));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& part_num_tensor = context->input(2);
    int new_num_part = part_num_tensor.flat<int32>()(0);

    // We always return the input ref.
    context->forward_ref_input_to_ref_output(0, 0);
    if (new_num_part == num_partitions_) return;
    {
      mutex_lock l(*context->input_ref_mutex(0));
      const Tensor& old_lhs = context->mutable_input(0, /* lock_held */ true);

      const Tensor& rhs = context->input(1);
      TensorShape new_shape = rhs.shape();
      int shard_unit = rhs.shape().dim_size(0) / new_num_part;
      int remainder = rhs.shape().dim_size(0) % new_num_part;
      int total_shard_unit = 0;

      if (num_partitions_ == 1) {
        VLOG(1) << "partition num is 1. Early return.";
        return;
      }
      if (device_id_ >= new_num_part) {
        VLOG(1) << "No need to reassign. Early return.";
        return;
      }

      if (new_num_part > num_partitions_) {
        calulate_part_num_and_offset(rhs, remainder, partition_id_, shard_unit,
                                     total_shard_unit);
        VLOG(1) << "SCALING UP: " << new_shape.dim_size(0) << " origin shape "
                << rhs.shape().dim_size(0) << " offset " << total_shard_unit;
      } else {
        if (partition_id_ > device_id_) {
          calulate_part_num_and_offset(rhs, remainder, device_id_, shard_unit,
                                       total_shard_unit);
        } else {
          calulate_part_num_and_offset(rhs, remainder, partition_id_,
                                       shard_unit, total_shard_unit);
        }
        VLOG(1) << "SCALING DOWN: " << new_shape.dim_size(0) << " origin shape "
                << rhs.shape().dim_size(0) << " offset " << total_shard_unit;
      }
      new_shape.set_dim(0, shard_unit);
      // Otherwise, create a new persistent tensor whose shape matches the
      // right hand side, hand off to lhs and copy the rhs into it.
      PersistentTensor copy;
      Tensor* copyTensor = nullptr;
      AllocatorAttributes attr;
      OP_REQUIRES_OK(context,
                     context->allocate_persistent(old_lhs.dtype(), new_shape,
                                                  &copy, &copyTensor, attr));
      // We track memory of variables in variable ops instead of in this
      // assign op.
      context->clear_recorded_memory();
      context->replace_ref_input(0, *copyTensor, /* lock_held */ true);
      if (use_exclusive_lock_) {
        Copy(context, copyTensor, rhs, new_num_part, total_shard_unit);
        return;
      }
      // The tensor has already been initialized and the right hand side
      // matches the left hand side's shape. We have been told to do the
      // copy outside the lock.
      Copy(context, copyTensor, rhs, new_num_part, total_shard_unit);
    }
  }

 private:
  void calulate_part_num_and_offset(const Tensor& rhs, int remainder,
                                    int part_id, int& shard_unit,
                                    int& total_shard_unit) {
    {
      int local_shard_unit = shard_unit;
      if (remainder != 0) {
        if (remainder > part_id) {
          shard_unit += 1;
        }
        for (int i = 0; i < part_id; ++i) {
          if (remainder-- > i) {
            total_shard_unit += (local_shard_unit + 1);
          } else {
            total_shard_unit += local_shard_unit;
          }
        }
      } else {
        for (int i = 0; i < part_id; ++i) {
          total_shard_unit += shard_unit;
        }
      }
      for (int j = 1; j < rhs.dims(); j++) {
        total_shard_unit *= rhs.dim_size(j);
      }
    }
  }

  void Copy(OpKernelContext* context, Tensor* output, const Tensor& rhs,
            int new_partition_nums, int offset) {
    if (new_partition_nums > num_partitions_) {
      functor::CustomScale<Device, T> copy;
      copy(context->eigen_device<Device>(), output->flat<T>(), rhs.flat<T>(),
           partition_id_, new_partition_nums, offset);
    } else {
      functor::CustomScale<Device, T> copy;
      copy(context->eigen_device<Device>(), output->flat<T>(), rhs.flat<T>(),
           partition_id_, new_partition_nums, offset);
    }
  }

 private:
  bool use_exclusive_lock_;
  int partition_id_;
  int device_id_;
  int num_partitions_;
};

typedef Eigen::ThreadPoolDevice CPUDevice;

#define REGISTER_KERNELS(type)                                             \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name(::des::kReAssign).Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      ReAssignOp<CPUDevice, type>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#if GOOGLE_CUDA
typedef Eigen::GpuDevice GPUDevice;
#define REGISTER_GPU_KERNELS(type)                                         \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name(::des::kReAssign).Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      ReAssignOp<GPUDevice, type>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
TF_CALL_int64(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS
#endif  // end GOOGLE_CUDA

template <typename Device, typename T>
class ReAssignResourceOp : public OpKernel {
 public:
  explicit ReAssignResourceOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("T", &dtype_));
    OP_REQUIRES_OK(context, context->GetAttr("partition_id", &partition_id_));
    OP_REQUIRES_OK(context, context->GetAttr("device_id", &device_id_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("partition_nums", &num_partitions_));
  }

  void Compute(OpKernelContext* context) override {
    OP_REQUIRES(context, dtype_ == context->input(1).dtype(),
                errors::InvalidArgument(
                    "Variable and value dtypes don't match; respectively, ",
                    DataTypeString(dtype_), " and ",
                    DataTypeString(context->input(1).dtype())));

    const Tensor& part_num_tensor = context->input(2);
    int new_num_part = part_num_tensor.flat<int32>()(0);
    Var* variable = nullptr;

    const Tensor& value = context->input(1);

    int shard_unit = value.shape().dim_size(0) / new_num_part;
    int remainder = value.shape().dim_size(0) % new_num_part;
    int total_shard_unit = 0;

    if (num_partitions_ == 1) {
      VLOG(1) << "partition num is 1. Early return.";
      return;
    }
    if (device_id_ >= new_num_part) {
      VLOG(1) << "No need to reassign. Early return.";
      return;
    }

    if (new_num_part == num_partitions_) {
      VLOG(1) << "partition num has not changed. Early return.";
      return;
    }
    OP_REQUIRES_OK(context,
                   LookupOrCreateResource<Var>(
                       context, HandleFromInput(context, 0), &variable,
                       [this, new_num_part, remainder, &total_shard_unit,
                        &shard_unit, &value](Var** ptr) {
                         *ptr = new Var(dtype_);
                         calulate_part_num(new_num_part, remainder, shard_unit,
                                           total_shard_unit);
                         *(*ptr)->tensor() = value.Slice(
                             total_shard_unit, total_shard_unit + shard_unit);
                         (*ptr)->is_initialized = true;
                         return Status::OK();
                       }));
    core::ScopedUnref s(variable);

    mutex_lock ml(*variable->mu());
    OP_REQUIRES(context, variable->tensor()->dtype() == dtype_,
                errors::InvalidArgument(
                    "Trying to assign variable with wrong dtype. Expected ",
                    DataTypeString(variable->tensor()->dtype()), " got ",
                    DataTypeString(dtype_)));

    calulate_part_num(new_num_part, remainder, shard_unit, total_shard_unit);
    *variable->tensor() =
        value.Slice(total_shard_unit, total_shard_unit + shard_unit);
    variable->is_initialized = true;
  }

 private:
  void calulate_part_num(int new_num_part, int remainder, int& shard_unit,
                         int& total_shard_unit) {
    auto calculate_func = [this, &remainder, &shard_unit,
                           &total_shard_unit](int part_id) {
      int local_shard_unit = shard_unit;
      if (remainder != 0) {
        if (remainder > part_id) {
          shard_unit += 1;
        }
        for (int i = 0; i < part_id; ++i) {
          if (remainder-- > i) {
            total_shard_unit += (local_shard_unit + 1);
          } else {
            total_shard_unit += local_shard_unit;
          }
        }
      } else {
        for (int i = 0; i < part_id; ++i) {
          total_shard_unit += shard_unit;
        }
      }
    };

    if (new_num_part > num_partitions_) {
      calculate_func(partition_id_);
    } else {
      if (partition_id_ > device_id_) {
        calculate_func(device_id_);
      } else {
        calculate_func(partition_id_);
      }
    }
  }

 private:
  DataType dtype_;
  int partition_id_;
  int device_id_;
  int num_partitions_;
};

#define REGISTER_KERNELS(type)                                                \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name(::des::kReAssignRes).Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      ReAssignResourceOp<Eigen::ThreadPoolDevice, type>);

TF_CALL_ALL_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#if GOOGLE_CUDA
typedef Eigen::GpuDevice GPUDevice;
#define REGISTER_GPU_KERNELS(type)                         \
  REGISTER_KERNEL_BUILDER(Name(::des::kReAssignRes)        \
                              .Device(DEVICE_GPU)          \
                              .TypeConstraint<type>("T")   \
                              .HostMemory("old_resource"), \
                          ReAssignResourceOp<GPUDevice, type>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
TF_CALL_int32(REGISTER_GPU_KERNELS);
TF_CALL_int64(REGISTER_GPU_KERNELS);
TF_CALL_int8(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS
#endif  // GOOGLE_CUDA

template <typename T>
class ReleaseResource : public OpKernel {};

}  // end namespace des
}  // end namespace tensorflow