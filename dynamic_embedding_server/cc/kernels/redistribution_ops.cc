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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif

#include "dynamic_embedding_server/include/ops/redistribution_functor.h"
#include "tensorflow/core/framework/embedding/embedding_var.h"
#include "tensorflow/core/framework/embedding/value_ptr.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace des {

template <typename TKey, typename TValue>
class KvResourceFilterOp : public OpKernel {
 public:
  explicit KvResourceFilterOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("partition_id", &partition_id_));
  }

  void Compute(OpKernelContext* ctx) override {
    EmbeddingVar<TKey, TValue>* embedding_var = nullptr;
    OP_REQUIRES_OK(
        ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &embedding_var));
    core::ScopedUnref unref_me(embedding_var);
    const Tensor& partition_num_tensor = ctx->input(1);
    int partition_num = partition_num_tensor.flat<int>()(0);

    std::vector<TKey> filtered_keys;
    std::vector<ValuePtr<TValue>*> value_ptr_list;
    int64 before_size = embedding_var->Size();
    OP_REQUIRES_OK(ctx,
                   embedding_var->GetShardedSnapshot(&filtered_keys, &value_ptr_list,
                                                     partition_id_, partition_num));
    int64 after_size = embedding_var->Size();
    VLOG(1) << " KvResourceFilterOp: " << embedding_var->Name()
            << " before_size " << before_size << " after_size " << after_size
            << " filtered_keys size: " << filtered_keys.size()
            << " partition_num: " << partition_num;
    Tensor* unneeded_ids_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {filtered_keys.size()},
                                             &unneeded_ids_tensor));
    Tensor* unneeded_value_tensor = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(
                       1, {filtered_keys.size(), embedding_var->ValueLen()},
                       &unneeded_value_tensor));
    Tensor* version_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(2, {filtered_keys.size()}, &version_tensor));
    Tensor* freq_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(3, {filtered_keys.size()}, &freq_tensor));
    if (filtered_keys.size() == 0) {
      return;
    }
    auto unneeded_ids = unneeded_ids_tensor->flat<TKey>().data();
    auto unneeded_value = unneeded_value_tensor->flat<TValue>().data();
    auto versions = version_tensor->flat<int64>().data();
    auto freq = freq_tensor->flat<int64>().data();
    embedding_var->ExportAndRemove(unneeded_ids, unneeded_value, versions, freq,
                                   filtered_keys, value_ptr_list);
  }

 private:
  int partition_id_;
};

#define REGISTER_CPU_KERNELS(key_type, value_type)                  \
  REGISTER_KERNEL_BUILDER(Name("KvResourceFilter")                  \
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
    int64 before_size = embedding_var->Size();
    // part 0 is itself, skipped.
    for (int i = 1; i < partition_nums_; ++i) {
      const Tensor& import_ids_tensor = ctx->input(1 + i);
      auto* import_ids = import_ids_tensor.flat<TKey>().data();
      int64 N = import_ids_tensor.NumElements();
      if (N == 0) continue;
      const Tensor& import_values_tensor = ctx->input(1 + partition_nums_ + i);
      auto* import_values = import_values_tensor.flat<float>().data();

      const Tensor& import_versions_tensor =
          ctx->input(1 + partition_nums_ * 2 + i);
      auto* import_versions = import_versions_tensor.flat<int64>().data();
      const Tensor& import_freqs_tensor =
          ctx->input(1 + partition_nums_ * 3 + i);
      auto* import_freqs = import_freqs_tensor.flat<int64>().data();
      OP_REQUIRES_OK(ctx,
                     embedding_var->RestoreFromKeysAndValues(
                         N, partition_id_, (partition_nums_ + 1), import_ids,
                         import_values, import_versions, import_freqs));
    }
    int64 after_size = embedding_var->Size();
    VLOG(1) << " KvResourceMulImport: " << embedding_var->Name()
            << " after_size " << after_size << " before_size " << before_size;
  }

 private:
  int partition_id_;
  int partition_nums_;
};

#define REGISTER_CPU_KERNELS(key_type, value_type)                  \
  REGISTER_KERNEL_BUILDER(Name("KvResourceMulImport")               \
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

      TensorShape new_shape = old_lhs.shape();
      int shard_unit = rhs.shape().dim_size(0) / new_num_part;

      if (num_partitions_ == 1) {
        LOG(INFO) << "partition num is 1. Early return.";
        return;
      }
      if (new_num_part > num_partitions_) {
        if (partition_id_ == (new_num_part - 1)) {
          new_shape.set_dim(
              0, (rhs.shape().dim_size(0) - shard_unit * (new_num_part - 1)));
        } else {
          new_shape.set_dim(0, shard_unit);
        }
        LOG(INFO) << "scale up new shape " << new_shape.dim_size(0)
                  << " original full shape " << rhs.shape().dim_size(0);
      } else {
        shard_unit = rhs.shape().dim_size(0) / new_num_part;
        if (partition_id_ == (new_num_part - 1)) {
          new_shape.set_dim(
              0, (rhs.shape().dim_size(0) - shard_unit * (new_num_part - 1)));
        } else {
          new_shape.set_dim(0, shard_unit);
        }
        LOG(INFO) << "scale down new shape " << new_shape.dim_size(0)
                  << " original full shape " << rhs.shape().dim_size(0);
      }

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
        Copy(context, copyTensor, rhs, new_num_part,
             shard_unit * partition_id_);
        return;
      }
      // The tensor has already been initialized and the right hand side
      // matches the left hand side's shape. We have been told to do the
      // copy outside the lock.
      Copy(context, copyTensor, rhs, new_num_part, shard_unit * partition_id_);
    }
  }

 private:
  void Copy(OpKernelContext* context, Tensor* output, const Tensor& rhs,
            int new_partition_nums, int offset) {
    if (new_partition_nums > num_partitions_) {
      functor::CustomScale<Device, T> copy;
      copy(context->eigen_device<Device>(), output->flat<T>(), rhs.flat<T>(),
           partition_id_, new_partition_nums, offset);
    } else {
      if (partition_id_ == new_partition_nums) return;
      functor::CustomScale<Device, T> copy;
      copy(context->eigen_device<Device>(), output->flat<T>(), rhs.flat<T>(),
           partition_id_, new_partition_nums, offset);
    }
  }

 private:
  bool use_exclusive_lock_;
  int partition_id_;
  int num_partitions_;
};

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

#define REGISTER_KERNELS(type)                                       \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("ReAssign").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      ReAssignOp<CPUDevice, type>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU_KERNELS(type)                                   \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("ReAssign").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      ReAssignOp<GPUDevice, type>);                                  \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("ReAssign").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      ReAssignOp<GPUDevice, type>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
TF_CALL_int64(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS
#endif  // end GOOGLE_CUDA || TENSORFLOW_USE_ROCM

template <typename Device, typename T>
class ReAssignResourceOp : public OpKernel {
 public:
  explicit ReAssignResourceOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("T", &dtype_));
    OP_REQUIRES_OK(context, context->GetAttr("partition_id", &partition_id_));
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

    core::RefCountPtr<Var> variable;
    const Tensor& value = context->input(1);

    Tensor slice_tensor;
    int shard_unit = value.shape().dim_size(0) / new_num_part;

    if (num_partitions_ == 1) {
      LOG(INFO) << "partition num is 1. Early return.";
      return;
    }

    if (new_num_part == num_partitions_) {
      LOG(INFO) << "partition num has not changed. Early return.";
      return;
    } else if (new_num_part > num_partitions_) {
      if (partition_id_ == (new_num_part - 1)) {
        slice_tensor =
            value.Slice(partition_id_ * shard_unit, value.shape().dim_size(0));
      } else {
        slice_tensor = value.Slice(partition_id_ * shard_unit,
                                   (partition_id_ + 1) * shard_unit);
      }
      LOG(INFO) << "scale up new shape " << slice_tensor.dim_size(0)
                << " original full shape " << value.shape().dim_size(0);
    } else {
      if (partition_id_ >= new_num_part) {
        LOG(INFO) << "No need to scale";
        return;
      } else if (partition_id_ == (new_num_part - 1)) {
        slice_tensor =
            value.Slice(partition_id_ * shard_unit, value.shape().dim_size(0));
      } else {
        slice_tensor = value.Slice(partition_id_ * shard_unit,
                                   (partition_id_ + 1) * shard_unit);
      }
      LOG(INFO) << "scale down new shape " << slice_tensor.dim_size(0)
                << " original full shape " << value.shape().dim_size(0);
    }

    OP_REQUIRES_OK(context, LookupOrCreateResource<Var>(
                                context, HandleFromInput(context, 0), &variable,
                                [this, &slice_tensor](Var** ptr) {
                                  *ptr = new Var(dtype_);
                                  *(*ptr)->tensor() = slice_tensor;
                                  (*ptr)->is_initialized = true;
                                  return Status::OK();
                                }));
    mutex_lock ml(*variable->mu());
    OP_REQUIRES(context, variable->tensor()->dtype() == dtype_,
                errors::InvalidArgument(
                    "Trying to assign variable with wrong dtype. Expected ",
                    DataTypeString(variable->tensor()->dtype()), " got ",
                    DataTypeString(dtype_)));
    if (variable->copy_on_read_mode.load()) {
      PersistentTensor unused;
      Tensor* tmp;
      AllocatorAttributes attr;
      attr.set_gpu_compatible(true);
      attr.set_nic_compatible(true);
      OP_REQUIRES_OK(context, context->allocate_persistent(
                                  value.dtype(), slice_tensor.shape(), &unused,
                                  &tmp, attr));
      functor::CustomDenseUpdate<Device, T> copy_functor;
      copy_functor(context->eigen_device<Device>(), tmp->flat<T>(),
                   slice_tensor.flat<T>());
      *variable->tensor() = *tmp;
    } else {
      *variable->tensor() = slice_tensor;
    }
    variable->is_initialized = true;
  }

 private:
  DataType dtype_;
  int partition_id_;
  int num_partitions_;
};

#define REGISTER_KERNELS(type)                                               \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("ReAssignResource").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      ReAssignResourceOp<Eigen::ThreadPoolDevice, type>);

TF_CALL_ALL_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU_KERNELS(type)                         \
  REGISTER_KERNEL_BUILDER(Name("ReAssignResource")         \
                              .Device(DEVICE_GPU)          \
                              .TypeConstraint<type>("T")   \
                              .HostMemory("old_resource"), \
                          AssignVariableOp<GPUDevice, type>);

TF_CALL_GPU_ALL_TYPES(REGISTER_GPU_KERNELS);
TF_CALL_int64(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // end namespace des
}  // end namespace tensorflow