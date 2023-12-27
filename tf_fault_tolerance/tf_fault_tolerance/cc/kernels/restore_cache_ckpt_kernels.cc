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
==============================================================================*/

#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"

#include "tf_fault_tolerance/cc/kernels/restore_cache_ckpt_kernels.h"
#include "tf_fault_tolerance/cc/utils/cache_ckpt/naming.h"
#include "tf_fault_tolerance/cc/utils/platform/file.h"

namespace tensorflow {

//------------------------------------------------------------------------------
// functions of CheckLocalCacheCKPTOp

CheckLocalCacheCKPTOp::CheckLocalCacheCKPTOp(OpKernelConstruction* context)
  : CacheCKPTOp(context) {
  OP_REQUIRES_OK(context, context->GetAttr("container", &container_));
  OP_REQUIRES_OK(context, context->GetAttr("shared_name", &name_));
}

CheckLocalCacheCKPTOp::~CheckLocalCacheCKPTOp() {}

void CheckLocalCacheCKPTOp::Compute(OpKernelContext* context) {
  const Tensor& ckpt_path_prefix_tensor = context->input(0);
  const Tensor& shard_tensor = context->input(1);
  const Tensor& num_shards_tensor = context->input(2);

  ValidateInputs(context, ckpt_path_prefix_tensor, shard_tensor, \
                 num_shards_tensor);
  if (!context->status().ok()) {
    return;
  }

  const std::string& ckpt_path_prefix = \
    ckpt_path_prefix_tensor.scalar<tstring>()();
  const int shard = shard_tensor.scalar<int>()();
  const int num_shards = num_shards_tensor.scalar<int>()();

  Tensor* exist_cache_ckpt_tensor;
  OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}),
                                                   &exist_cache_ckpt_tensor));
  Tensor* cache_ckpt_handle_tensor;
  OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({}),
                                                   &cache_ckpt_handle_tensor));
  // Create resource handle.
  ResourceHandle handle = \
    MakeResourceHandle<CacheCKPTVar>(context, container_, name_);

  // Try to get cache ckpt.
  const std::string& ckpt_key = \
    GenerateCKPTKey(ckpt_path_prefix, shard, num_shards);
  bool exist_cache_ckpt = false;
  OP_REQUIRES_OK(context, TryToGetCacheCKPT(context, ckpt_key, handle,
                                            exist_cache_ckpt));
  exist_cache_ckpt_tensor->scalar<bool>()() = exist_cache_ckpt;
  cache_ckpt_handle_tensor->scalar<ResourceHandle>()() = handle;

  if (exist_cache_ckpt) {
    LOG(INFO) << "CacheCKPT: Use cache ckpt on local storage.";
  } else {
    LOG(WARNING) << "CheckLocalCacheCKPTOp: Local cache ckpt does not exist.";
  }
}

void CheckLocalCacheCKPTOp::ValidateInputs(OpKernelContext* context,
                                           const Tensor& ckpt_path_prefix,
                                           const Tensor& shard,
                                           const Tensor& num_shards) {
  OP_REQUIRES(context, TensorShapeUtils::IsScalar(ckpt_path_prefix.shape()),
              errors::InvalidArgument("ckpt_path_prefix is not a scalar: ",
                                      ckpt_path_prefix.shape().DebugString()));
  OP_REQUIRES(context, TensorShapeUtils::IsScalar(shard.shape()),
              errors::InvalidArgument("shard is not a scalar: ",
                                      shard.shape().DebugString()));
  OP_REQUIRES(context, TensorShapeUtils::IsScalar(num_shards.shape()),
              errors::InvalidArgument("num_shards is not a scalar: ",
                                      num_shards.shape().DebugString()));
}

Status CheckLocalCacheCKPTOp::TryToGetCacheCKPT(OpKernelContext* ctx,
                                                const std::string& ckpt_key,
                                                ResourceHandle& handle,
                                                bool& is_exist_cache) {
  Tensor meta_tensor, data_tensor;
  TF_RETURN_IF_ERROR(ctx->allocate_temp(DT_STRING, TensorShape({}),
                                        &meta_tensor));
  TF_RETURN_IF_ERROR(ctx->allocate_temp(DT_STRING, TensorShape({}),
                                        &data_tensor));

  std::string& meta_file = meta_tensor.scalar<tstring>()();
  std::string& data_file = data_tensor.scalar<tstring>()();

  CacheCKPTManager* cache_ckpt_mgr = CacheCKPTManager::GetInstance();
  is_exist_cache = \
    cache_ckpt_mgr->TryToGetCacheCKPT(ckpt_key, false, meta_file, data_file);

  if (is_exist_cache) {
    core::RefCountPtr<CacheCKPTVar> cache_ckpt_var;
    TF_RETURN_IF_ERROR(GetOrCreateCacheCKPTVar(ctx, handle, cache_ckpt_var));
    mutex_lock ml(*(cache_ckpt_var->mu()));
    cache_ckpt_var->update(meta_tensor, data_tensor);
  }

  return Status::OK();
}

REGISTER_KERNEL_BUILDER(Name("CheckLocalCacheCKPT").Device(DEVICE_CPU),
                        CheckLocalCacheCKPTOp);

//------------------------------------------------------------------------------
// functions of GetRemoteCacheCKPTOp
GetRemoteCacheCKPTOp::GetRemoteCacheCKPTOp(OpKernelConstruction* context)
  : CacheCKPTOp(context) {
  OP_REQUIRES_OK(context, context->GetAttr("container", &container_));
  OP_REQUIRES_OK(context, context->GetAttr("shared_name", &name_));
  OP_REQUIRES_OK(context, context->GetAttr("is_merged_meta", &is_merged_meta_));
  OP_REQUIRES_OK(context,
                 context->GetAttr("ckpt_storage_type", &ckpt_storage_type_));
}

GetRemoteCacheCKPTOp::~GetRemoteCacheCKPTOp() {}

void GetRemoteCacheCKPTOp::Compute(OpKernelContext* context) {
  const Tensor& ckpt_path_prefix_tensor = context->input(0);
  const Tensor& cache_path_tensor = context->input(1);
  const Tensor& exist_cache_ckpt_tensor = context->input(2);
  const Tensor& ckpt_key_tensor = context->input(3);
  const Tensor& ckpt_meta_tensor = context->input(4);
  const Tensor& ckpt_data_tensor = context->input(5);

  ValidateInputs(context, ckpt_path_prefix_tensor, cache_path_tensor,
                 exist_cache_ckpt_tensor, ckpt_key_tensor, ckpt_meta_tensor,
                 ckpt_data_tensor);
  if (!context->status().ok()) {
    return;
  }

  const std::string& ckpt_path_prefix = \
    ckpt_path_prefix_tensor.scalar<tstring>()();
  const std::string& cache_path = cache_path_tensor.scalar<tstring>()();
  const bool exist_cache_ckpt = exist_cache_ckpt_tensor.scalar<bool>()();
  const std::string& ckpt_key = ckpt_key_tensor.scalar<tstring>()();
  std::string ckpt_filename_prefix, unused;
  ParsePOSIXFilePath(ckpt_path_prefix, unused, ckpt_filename_prefix);
  std::string ckpt_filename_prefix_without_global_step = \
    CKPTPathPrefixWithoutGlobalStep(ckpt_filename_prefix);

  Tensor* exist_cache_ckpt_out_tensor;
  OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}),
                                     &exist_cache_ckpt_out_tensor));
  Tensor* ckpt_handle_tensor;
  OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({}),
                                                   &ckpt_handle_tensor));
  // Create resource handle.
  ResourceHandle handle = \
    MakeResourceHandle<CacheCKPTVar>(context, container_, name_);
  ckpt_handle_tensor->scalar<ResourceHandle>()() = handle;

  bool exist_cache_ckpt_out = false;
  const std::string& ckpt_meta_file = ckpt_meta_tensor.scalar<tstring>()();
  const std::string& ckpt_data_file = ckpt_data_tensor.scalar<tstring>()();
  if (exist_cache_ckpt) {
    // Generate local cache ckpt.
    OP_REQUIRES_OK(context,
                   GenerateLocalCacheCKPT(context, handle, ckpt_key, cache_path,
                     ckpt_filename_prefix_without_global_step, ckpt_meta_file,
                     ckpt_data_file, exist_cache_ckpt_out));
  } else {
    LOG(WARNING) << "GetRemoteCacheCKPTOp: Remote cache ckpt does not exist.";
    Env* env = Env::Default();
    env->DeleteFile(ckpt_meta_file);
    env->DeleteFile(ckpt_data_file);
  }

  if (exist_cache_ckpt_out) {
    LOG(INFO) << "CacheCKPT: Get remote cache ckpt success.";
  } else {
    LOG(WARNING) << "GetRemoteCacheCKPTOp: Failed to fetch remote cache ckpt.";
  }
  exist_cache_ckpt_out_tensor->scalar<bool>()() = exist_cache_ckpt_out;
}

void GetRemoteCacheCKPTOp::ValidateInputs(OpKernelContext* context,
                                          const Tensor& ckpt_path_prefix,
                                          const Tensor& cache_path,
                                          const Tensor& exist_cache_ckpt,
                                          const Tensor& ckpt_key,
                                          const Tensor& ckpt_meta,
                                          const Tensor& ckpt_data) {
  OP_REQUIRES(context, TensorShapeUtils::IsScalar(ckpt_path_prefix.shape()),
              errors::InvalidArgument("ckpt_path_prefix is not a scalar: ",
                                      ckpt_path_prefix.shape().DebugString()));
  OP_REQUIRES(context, TensorShapeUtils::IsScalar(cache_path.shape()),
              errors::InvalidArgument("cache_path is not a scalar: ",
                                      cache_path.shape().DebugString()));
  OP_REQUIRES(context, TensorShapeUtils::IsScalar(exist_cache_ckpt.shape()),
              errors::InvalidArgument("exist_cache_ckpt is not a scalar: ",
                                      exist_cache_ckpt.shape().DebugString()));
  OP_REQUIRES(context, TensorShapeUtils::IsScalar(ckpt_key.shape()),
              errors::InvalidArgument("ckpt_key is not a scalar: ",
                                      ckpt_key.shape().DebugString()));
  OP_REQUIRES(context, TensorShapeUtils::IsScalar(ckpt_meta.shape()),
              errors::InvalidArgument("ckpt_meta is not a scalar: ",
                                      ckpt_meta.shape().DebugString()));
  OP_REQUIRES(context, TensorShapeUtils::IsScalar(ckpt_data.shape()),
              errors::InvalidArgument("ckpt_data is not a scalar: ",
                                      ckpt_data.shape().DebugString()));
}

Status GetRemoteCacheCKPTOp::GenerateLocalCacheCKPT(
                               OpKernelContext* context,
                               ResourceHandle& handle,
                               const std::string& ckpt_key,
                               const std::string& cache_path,
                               const std::string& ckpt_filename_prefix,
                               const std::string& meta_file,
                               const std::string& data_file,
                               bool& exist_cache_ckpt) {
  struct CacheCKPTManagerParams params;
  params.is_local_ckpt = true;
  params.is_merged_meta = is_merged_meta_;
  params.storage_type = ckpt_storage_type_;
  params.cache_path = cache_path;
  params.ckpt_filename_prefix = ckpt_filename_prefix;

  CacheCKPTManager* cache_ckpt_mgr = CacheCKPTManager::GetInstance();
  cache_ckpt_mgr->UpdateCacheCKPT(ckpt_key, meta_file, data_file, true, params);

  Tensor meta_out_tensor, data_out_tensor;
  TF_RETURN_IF_ERROR(context->allocate_temp(DT_STRING, TensorShape({}),
                                            &meta_out_tensor));
  TF_RETURN_IF_ERROR(context->allocate_temp(DT_STRING, TensorShape({}),
                                            &data_out_tensor));
  std::string& meta_file_out = meta_out_tensor.scalar<tstring>()();
  std::string& data_file_out = data_out_tensor.scalar<tstring>()();
  exist_cache_ckpt = \
    cache_ckpt_mgr->TryToGetCacheCKPT(ckpt_key, false, meta_file_out,
                                      data_file_out);

  core::RefCountPtr<CacheCKPTVar> cache_ckpt_var;
  TF_RETURN_IF_ERROR(GetOrCreateCacheCKPTVar(context, handle, cache_ckpt_var));
  mutex_lock ml(*(cache_ckpt_var->mu()));
  cache_ckpt_var->update(meta_out_tensor, data_out_tensor);

  return Status::OK();
}

REGISTER_KERNEL_BUILDER(Name("GetRemoteCacheCKPT").Device(DEVICE_CPU),
                        GetRemoteCacheCKPTOp);

//------------------------------------------------------------------------------
// functions of RepatriateRemoteCacheCKPTOp

RepatriateRemoteCacheCKPTOp::RepatriateRemoteCacheCKPTOp(
                               OpKernelConstruction* context)
  : CacheCKPTOp(context) {
  OP_REQUIRES_OK(context, context->GetAttr("output_is_path", &output_is_path_));
}

RepatriateRemoteCacheCKPTOp::~RepatriateRemoteCacheCKPTOp() {}

void RepatriateRemoteCacheCKPTOp::Compute(OpKernelContext* context) {
  const Tensor& ckpt_path_prefix_tensor = context->input(0);
  const Tensor& shard_tensor = context->input(1);
  const Tensor& num_shards_tensor = context->input(2);
  ValidateInputs(context, ckpt_path_prefix_tensor, shard_tensor,
                 num_shards_tensor);
  if (!context->status().ok()) {
    return;
  }

  const std::string& ckpt_path_prefix = \
    ckpt_path_prefix_tensor.scalar<tstring>()();
  const int& shard = shard_tensor.scalar<int>()();
  const int& num_shards = num_shards_tensor.scalar<int>()();
  const std::string ckpt_key = \
    GenerateCKPTKey(ckpt_path_prefix, shard, num_shards);

  Tensor* exist_cache_ckpt_tensor;
  OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}),
                                                   &exist_cache_ckpt_tensor));
  Tensor* ckpt_key_tensor;
  OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({}),
                                                   &ckpt_key_tensor));
  Tensor* ckpt_meta_tensor;
  OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape({}),
                                                   &ckpt_meta_tensor));
  Tensor* ckpt_data_tensor;
  OP_REQUIRES_OK(context, context->allocate_output(3, TensorShape({}),
                                                   &ckpt_data_tensor));

  // File ckpt_key_tensor.
  ckpt_key_tensor->scalar<tstring>()() = ckpt_key;

  // Fill ckpt_tensor
  bool exist_cache_ckpt = \
    FillCacheCKPTOutput(ckpt_key, *ckpt_meta_tensor, *ckpt_data_tensor);

  // Fill exist_cache_ckpt_tensor
  exist_cache_ckpt_tensor->scalar<bool>()() = exist_cache_ckpt;
}

void RepatriateRemoteCacheCKPTOp::ValidateInputs(OpKernelContext* context,
                                                 const Tensor& ckpt_path_prefix,
                                                 const Tensor& shard,
                                                 const Tensor& num_shards) {
  OP_REQUIRES(context, TensorShapeUtils::IsScalar(ckpt_path_prefix.shape()),
              errors::InvalidArgument("ckpt_path_prefix is not a scalar: ",
                                      ckpt_path_prefix.shape().DebugString()));
  OP_REQUIRES(context, TensorShapeUtils::IsScalar(shard.shape()),
              errors::InvalidArgument("shard is not a scalar: ",
                                      shard.shape().DebugString()));
  OP_REQUIRES(context, TensorShapeUtils::IsScalar(num_shards.shape()),
              errors::InvalidArgument("num_shards is not a scalar: ",
                                      num_shards.shape().DebugString()));
}

bool RepatriateRemoteCacheCKPTOp::FillCacheCKPTOutput(
                                    const std::string& ckpt_key,
                                    Tensor& ckpt_meta, Tensor& ckpt_data) {
  CacheCKPTManager* ckpt_mgr = CacheCKPTManager::GetInstance();

  if (output_is_path_) {
    std::string& meta_path = ckpt_meta.scalar<tstring>()();
    std::string& data_path = ckpt_data.scalar<tstring>()();

    return ckpt_mgr->TryToGetCacheCKPT(ckpt_key, true, meta_path, data_path);
  }

  return ckpt_mgr->TryToGetCacheCKPT(ckpt_key, ckpt_meta, ckpt_data);
}

REGISTER_KERNEL_BUILDER(Name("RepatriateRemoteCacheCKPT").Device(DEVICE_CPU),
                        RepatriateRemoteCacheCKPTOp);

//------------------------------------------------------------------------------
// functions of LoadCKPTFromFilePathOp

LoadCKPTFromFilePathOp::LoadCKPTFromFilePathOp(OpKernelConstruction* context)
  : CacheCKPTOp(context) {
  OP_REQUIRES_OK(context, context->GetAttr("container", &container_));
  OP_REQUIRES_OK(context, context->GetAttr("shared_name", &name_));
  OP_REQUIRES_OK(context, context->GetAttr("output_is_path", &output_is_path_));
}

LoadCKPTFromFilePathOp::~LoadCKPTFromFilePathOp() {}

void LoadCKPTFromFilePathOp::Compute(OpKernelContext* context) {
  const Tensor& ckpt_path_prefix_tensor = context->input(0);
  const Tensor& shard_tensor = context->input(1);
  const Tensor& num_shards_tensor = context->input(2);

  ValidateInputs(context, ckpt_path_prefix_tensor, shard_tensor,
                 num_shards_tensor);
  if (!context->status().ok()) {
    return;
  }

  const std::string& ckpt_path_prefix = \
    ckpt_path_prefix_tensor.scalar<tstring>()();
  const int shard = shard_tensor.scalar<int>()();
  const int num_shards = num_shards_tensor.scalar<int>()();

  Tensor* ckpt_handle_tensor;
  OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}),
                                                   &ckpt_handle_tensor));
  ResourceHandle handle = \
    MakeResourceHandle<CacheCKPTVar>(context, container_, name_);
  core::RefCountPtr<CacheCKPTVar> cache_ckpt_var;
  OP_REQUIRES_OK(context,
                 GetOrCreateCacheCKPTVar(context, handle, cache_ckpt_var));
  ckpt_handle_tensor->scalar<ResourceHandle>()() = handle;

  Tensor ckpt_meta_tensor, ckpt_data_tensor;
  OP_REQUIRES_OK(context, context->allocate_temp(DT_STRING, TensorShape({}),
                                                 &ckpt_meta_tensor));
  OP_REQUIRES_OK(context, context->allocate_temp(DT_STRING, TensorShape({}),
                                                 &ckpt_data_tensor));
  if (output_is_path_) {
    ckpt_meta_tensor.scalar<tstring>()() = ckpt_path_prefix;
    ckpt_data_tensor.scalar<tstring>()() = ckpt_path_prefix;
  } else {
    const std::string ckpt_meta_file_path = CKPTMetaFilename(ckpt_path_prefix);
    const std::string ckpt_data_file_path = \
      CKPTDataFilename(ckpt_path_prefix, shard, num_shards);
    Env* env = Env::Default();
    OP_REQUIRES_OK(context, ReadFileToString(env, ckpt_meta_file_path,
                              ckpt_meta_tensor.scalar<tstring>().data()));
    OP_REQUIRES_OK(context, ReadFileToString(env, ckpt_data_file_path,
                              ckpt_data_tensor.scalar<tstring>().data()));
  }
  mutex_lock ml(*(cache_ckpt_var->mu()));
  cache_ckpt_var->update(ckpt_meta_tensor, ckpt_data_tensor);
  LOG(INFO) << "CacheCKPT: Load ckpt from path: " << ckpt_path_prefix;
}

void LoadCKPTFromFilePathOp::ValidateInputs(OpKernelContext* context,
                                            const Tensor& ckpt_path_prefix,
                                            const Tensor& shard,
                                            const Tensor& num_shards) {
  OP_REQUIRES(context, TensorShapeUtils::IsScalar(ckpt_path_prefix.shape()),
              errors::InvalidArgument("ckpt_path_prefix is not a scalar: ",
                                      ckpt_path_prefix.shape().DebugString()));
  OP_REQUIRES(context, TensorShapeUtils::IsScalar(shard.shape()),
                errors::InvalidArgument("shard is not a scalar: ",
                                        shard.shape().DebugString()));
  OP_REQUIRES(context, TensorShapeUtils::IsScalar(num_shards.shape()),
              errors::InvalidArgument("num_shards is not a scalar: ",
                                      num_shards.shape().DebugString()));
}

REGISTER_KERNEL_BUILDER(Name("LoadCKPTFromFilePath").Device(DEVICE_CPU),
                        LoadCKPTFromFilePathOp);

//------------------------------------------------------------------------------
// functions of UnPackCacheCKPTOp.

UnPackCacheCKPTResourceOp::UnPackCacheCKPTResourceOp(
                             OpKernelConstruction* context)
  :CacheCKPTOp(context) {}

UnPackCacheCKPTResourceOp::~UnPackCacheCKPTResourceOp() {}

void UnPackCacheCKPTResourceOp::Compute(OpKernelContext* context) {
  core::RefCountPtr<CacheCKPTVar> cache_ckpt_var;
  const ResourceHandle& handle = HandleFromInput(context, 0);
  OP_REQUIRES_OK(context,
                 GetOrCreateCacheCKPTVar(context, handle, cache_ckpt_var));
  Tensor* ckpt_meta_tensor;
  OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}),
                                                   &ckpt_meta_tensor));
  *ckpt_meta_tensor = *(cache_ckpt_var->meta_tensor());
  Tensor* ckpt_data_tensor;
  OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({}),
                                                   &ckpt_data_tensor));
  *ckpt_data_tensor = *(cache_ckpt_var->data_tensor());
}

REGISTER_KERNEL_BUILDER(Name("UnPackCacheCKPTResource").Device(DEVICE_CPU),
                        UnPackCacheCKPTResourceOp);

} // End of namespace tensorflow
