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

#include "tf_fault_tolerance/cc/common/cache_ckpt/cache_ckpt_manager.h"
#include "tf_fault_tolerance/cc/kernels/save_cache_ckpt_kernels.h"
#include "tf_fault_tolerance/cc/utils/cache_ckpt/naming.h"
#include "tf_fault_tolerance/cc/utils/platform/file.h"

namespace tensorflow {

//------------------------------------------------------------------------------
// functions of GenerCacheCKPTOp.

GenerateCacheCKPTOp::GenerateCacheCKPTOp(OpKernelConstruction* context)
  : CacheCKPTOp(context) {
  OP_REQUIRES_OK(context, context->GetAttr("is_merged_meta", &is_merged_meta_));
  OP_REQUIRES_OK(context,
                 context->GetAttr("ckpt_storage_type", &ckpt_storage_type_));
}

GenerateCacheCKPTOp::~GenerateCacheCKPTOp() {}

void GenerateCacheCKPTOp::Compute(OpKernelContext* context) {
  // ckpt_path_prefix_tensor contains global_step
  const Tensor& ckpt_path_prefix_tensor = context->input(0);
  const Tensor& cache_path_tensor = context->input(1);
  const Tensor& shard_tensor = context->input(2);
  const Tensor& num_shards_tensor = context->input(3);

  ValidateInputs(context, ckpt_path_prefix_tensor, cache_path_tensor,
                 shard_tensor, num_shards_tensor);
  if (!context->status().ok()) {
    return;
  }

  const std::string& ckpt_path_prefix = \
    ckpt_path_prefix_tensor.scalar<tstring>()();
  const std::string& cache_path = cache_path_tensor.scalar<tstring>()();
  const int shard = shard_tensor.scalar<int>()();
  const int num_shards = num_shards_tensor.scalar<int>()();
  const std::string& ckpt_meta_path = CKPTMetaFilename(ckpt_path_prefix);
  const std::string& ckpt_data_path = \
    CKPTDataFilename(ckpt_path_prefix, shard, num_shards);
  std::string ckpt_filename_prefix, unused;
  ParsePOSIXFilePath(ckpt_path_prefix, unused, ckpt_filename_prefix);

  // Generate ckpt key.
  Tensor* ckpt_key_tensor;
  OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}),
                                                   &ckpt_key_tensor));
  const std::string& ckpt_key =\
    GenerateCKPTKey(ckpt_path_prefix, shard, num_shards);
  ckpt_key_tensor->scalar<tstring>()() = ckpt_key;

  // Generate local cache ckpt.
  CacheCKPTManager* cache_ckpt_mgr = CacheCKPTManager::GetInstance();
  CreateLocalCacheCKPT(cache_ckpt_mgr, cache_path, ckpt_key,
                       ckpt_filename_prefix, ckpt_meta_path, ckpt_data_path);

  // Output ckpt.
  OP_REQUIRES_OK(context, OutputCKPT(context, cache_ckpt_mgr, ckpt_key));
}

void GenerateCacheCKPTOp::ValidateInputs(OpKernelContext* context,
                                         const Tensor& ckpt_path_prefix,
                                         const Tensor& cache_path,
                                         const Tensor& shard,
                                         const Tensor& num_shards) {
  OP_REQUIRES(context, TensorShapeUtils::IsScalar(ckpt_path_prefix.shape()),
              errors::InvalidArgument("ckpt_path_prefix is not a scalar: ",
                                      ckpt_path_prefix.shape().DebugString()));
  OP_REQUIRES(context, TensorShapeUtils::IsScalar(cache_path.shape()),
              errors::InvalidArgument("cache_path is not a scalar: ",
                                      cache_path.shape().DebugString()));
  OP_REQUIRES(context, TensorShapeUtils::IsScalar(shard.shape()),
              errors::InvalidArgument("shard is not a scalar: ",
                                      shard.shape().DebugString()));
  OP_REQUIRES(context, TensorShapeUtils::IsScalar(num_shards.shape()),
              errors::InvalidArgument("num_shards is not a scalar: ",
                                      num_shards.shape().DebugString()));
}

void GenerateCacheCKPTOp::CreateLocalCacheCKPT(
                            CacheCKPTManager* mgr,
                            const std::string& cache_path,
                            const std::string& ckpt_key,
                            const std::string& ckpt_filename_prefix,
                            const std::string& ckpt_meta_path,
                            const std::string& ckpt_data_path) {
  struct CacheCKPTManagerParams params;
  params.is_local_ckpt = true;
  params.is_merged_meta = is_merged_meta_;
  params.storage_type = ckpt_storage_type_;
  params.cache_path = cache_path;
  params.ckpt_filename_prefix = \
    CKPTPathPrefixWithoutGlobalStep(ckpt_filename_prefix);

  mgr->UpdateCacheCKPT(ckpt_key, ckpt_meta_path, ckpt_data_path, false, params);
}

Status GenerateCacheCKPTOp::OutputCKPT(OpKernelContext* ctx,
                                       CacheCKPTManager* mgr,
                                       const std::string& ckpt_key) {
  Tensor* meta_tensor;
  TF_RETURN_IF_ERROR(ctx->allocate_output(1, TensorShape({}), &meta_tensor));
  std::string& cache_meta_file = meta_tensor->scalar<tstring>()();

  Tensor* data_tensor;
  TF_RETURN_IF_ERROR(ctx->allocate_output(2, TensorShape({}), &data_tensor));
  std::string& cache_data_file = data_tensor->scalar<tstring>()();

  bool exist_cache_ckpt = \
    mgr->TryToGetCacheCKPT(ckpt_key, true, cache_meta_file, cache_data_file);
  if (!exist_cache_ckpt) {
    return errors::NotFound("Failed to get local cache ckpt.");
  }

  return Status::OK();
}

REGISTER_KERNEL_BUILDER(Name("GenerateCacheCKPT").Device(DEVICE_CPU),
                        GenerateCacheCKPTOp);

//------------------------------------------------------------------------------
// functions of BackupRemoteCacheCKPTOp.

BackupRemoteCacheCKPTOp::BackupRemoteCacheCKPTOp(OpKernelConstruction* context)
  : CacheCKPTOp(context) {
  OP_REQUIRES_OK(context, context->GetAttr("is_merged_meta", &is_merged_meta_));
  OP_REQUIRES_OK(context,
                 context->GetAttr("ckpt_storage_type", &ckpt_storage_type_));
}

BackupRemoteCacheCKPTOp::~BackupRemoteCacheCKPTOp() {}

void BackupRemoteCacheCKPTOp::Compute(OpKernelContext* context) {
  const Tensor& cache_path_tensor = context->input(0);
  const Tensor& ckpt_key_tensor = context->input(1);
  const Tensor& ckpt_meta_tensor = context->input(2);
  const Tensor& ckpt_data_tensor = context->input(3);

  ValidateInputs(context, cache_path_tensor, ckpt_key_tensor, ckpt_meta_tensor,
                 ckpt_data_tensor);
  if (!context->status().ok()) {
    return;
  }

  const std::string& meta_path = ckpt_meta_tensor.scalar<tstring>()();
  const std::string& data_path = ckpt_data_tensor.scalar<tstring>()();
  const std::string& cache_path = cache_path_tensor.scalar<tstring>()();
  const std::string& ckpt_key = ckpt_key_tensor.scalar<tstring>()();

  struct CacheCKPTManagerParams params;
  params.is_local_ckpt = false;
  params.is_merged_meta = is_merged_meta_;
  params.storage_type = ckpt_storage_type_;
  params.cache_path = cache_path;
  params.ckpt_filename_prefix = "model.cache_ckpt";

  CacheCKPTManager* cache_ckpt_mgr = CacheCKPTManager::GetInstance();
  cache_ckpt_mgr->UpdateCacheCKPT(ckpt_key, meta_path, data_path, true, params);
}

void BackupRemoteCacheCKPTOp::ValidateInputs(OpKernelContext* context,
                                             const Tensor& cache_path,
                                             const Tensor& ckpt_key,
                                             const Tensor& ckpt_meta,
                                             const Tensor& ckpt_data) {
  OP_REQUIRES(context, TensorShapeUtils::IsScalar(cache_path.shape()),
              errors::InvalidArgument("cache_path is not a scalar: ",
                                      cache_path.shape().DebugString()));
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

REGISTER_KERNEL_BUILDER(Name("BackupRemoteCacheCKPT").Device(DEVICE_CPU),
                        BackupRemoteCacheCKPTOp);
} // End of namespace tensorflow
