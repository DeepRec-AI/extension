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

#include "tf_fault_tolerance/cc/common/cache_ckpt/remote_cache_ckpt_manager.h"
#include "tf_fault_tolerance/cc/kernels/save_cache_ckpt_kernels.h"
#include "tf_fault_tolerance/cc/utils/cache_ckpt/naming.h"

namespace tensorflow {

//------------------------------------------------------------------------------
// functions of GenerCacheCKPTOp.

GenerateCacheCKPTOp::GenerateCacheCKPTOp(OpKernelConstruction* context)
  : OpKernel(context) {}

GenerateCacheCKPTOp::~GenerateCacheCKPTOp() {}

void GenerateCacheCKPTOp::Compute(OpKernelContext* context) {
  const Tensor& ckpt_prefix_tensor = context->input(0);
  const Tensor& cache_ckpt_path_tensor = context->input(1);
  const Tensor& shard_tensor = context->input(2);
  const Tensor& num_shards_tensor = context->input(3);

  ValidateInputs(context, ckpt_prefix_tensor, cache_ckpt_path_tensor,
                 shard_tensor, num_shards_tensor);
  if (!context->status().ok()) {
    return;
  }

  const std::string& ckpt_prefix = ckpt_prefix_tensor.scalar<tstring>()();
  const std::string& cache_ckpt_path = \
    cache_ckpt_path_tensor.scalar<tstring>()();
  const int shard = shard_tensor.scalar<int>()();
  const int num_shards = num_shards_tensor.scalar<int>()();

  string cache_ckpt_data_file;
  OP_REQUIRES_OK(context,
                 CreateLocalCacheCKPT(ckpt_prefix, cache_ckpt_path, shard,
                                      num_shards, cache_ckpt_data_file));

  Tensor* cache_ckpt_key_tensor;
  OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}),
                                                   &cache_ckpt_key_tensor));
  FillCacheCKPTKeyTensor(ckpt_prefix, shard, num_shards, cache_ckpt_key_tensor);

  Tensor* cache_ckpt_tensor;
  OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({}),
                                                   &cache_ckpt_tensor));
  FillCacheCKPTTensor(cache_ckpt_data_file, cache_ckpt_tensor);
}

Status GenerateCacheCKPTOp::FillCacheCKPTTensor(
                              const std::string& cache_ckpt_data_file,
                              Tensor* t) {
  Env* env = Env::Default();
  Status status = \
    ReadFileToString(env, cache_ckpt_data_file, t->scalar<tstring>().data());

  return status;
}

Status GenerateCacheCKPTOp::FillCacheCKPTKeyTensor(
                              const std::string& ckpt_path_prefix,
                              int32 shard_id, int32 num_shards, Tensor* t) {
  std::string cache_ckpt_key = \
    GenerateCacheCKPTKey(ckpt_path_prefix, shard_id, num_shards);
  t->scalar<tstring>()() = cache_ckpt_key;

  return Status::OK();
}

Status GenerateCacheCKPTOp::CreateLocalCacheCKPT(
                              const std::string& ckpt_path_prefix,
                              const std::string& cache_ckpt_path,
                              int32 shard_id, int32 num_shards,
                              std::string& cache_ckpt_data_file) {
  std::string ckpt_data_file = \
    CKPTDataFilename(ckpt_path_prefix, shard_id, num_shards);
  cache_ckpt_data_file = \
    CacheCKPTDataFilename(ckpt_path_prefix, cache_ckpt_path, shard_id,
                          num_shards);

  Env* env = Env::Default();
  // Try to create cache_ckpt_path directory.
  Status s = env->CreateDir(cache_ckpt_path);
  if (s != Status::OK() && s.code() != error::ALREADY_EXISTS) {
    return s;
  }

  // Delete old cache ckpt.
  // 1. delete data file.
  std::string cache_ckpt_data_files_pattern = \
    CacheCKPTDataFilenamePattern(ckpt_path_prefix, cache_ckpt_path, shard_id,
                                 num_shards);
  std::vector<std::string> cache_ckpt_filenames;
  TF_CHECK_OK(env->GetMatchingPaths(cache_ckpt_data_files_pattern,
                                    &cache_ckpt_filenames));
  for (auto filename : cache_ckpt_filenames) {
    env->DeleteFile(filename);
  }

  // 2. delete meta file.
  std::string cache_ckpt_meta_files_pattern = \
    CacheCKPTMetaFilenamePattern(ckpt_path_prefix, cache_ckpt_path);
  cache_ckpt_filenames.clear();
  TF_CHECK_OK(env->GetMatchingPaths(cache_ckpt_meta_files_pattern,
                                    &cache_ckpt_filenames));
  for (auto filename : cache_ckpt_filenames) {
    env->DeleteFile(filename);
  }

  s = env->CopyFile(ckpt_data_file, cache_ckpt_data_file);
  return s;
}

void GenerateCacheCKPTOp::ValidateInputs(OpKernelContext* context,
                                         const Tensor& ckpt_prefix,
                                         const Tensor& cache_ckpt_path,
                                         const Tensor& shard,
                                         const Tensor& num_shards) {
  OP_REQUIRES(context, TensorShapeUtils::IsScalar(ckpt_prefix.shape()),
              errors::InvalidArgument("ckpt_prefix is not a scalar: ",
                                      ckpt_prefix.shape().DebugString()));
  OP_REQUIRES(context, TensorShapeUtils::IsScalar(cache_ckpt_path.shape()),
              errors::InvalidArgument("cache_ckpt_path is not a scalar: ",
                                      cache_ckpt_path.shape().DebugString()));
  OP_REQUIRES(context, TensorShapeUtils::IsScalar(shard.shape()),
              errors::InvalidArgument("shard is not a scalar: ",
                                      shard.shape().DebugString()));
  OP_REQUIRES(context, TensorShapeUtils::IsScalar(num_shards.shape()),
              errors::InvalidArgument("num_shards is not a scalar: ",
                                      num_shards.shape().DebugString()));
}

REGISTER_KERNEL_BUILDER(Name("GenerateCacheCKPT").Device(DEVICE_CPU),
                        GenerateCacheCKPTOp);

//------------------------------------------------------------------------------
// functions of RecvRemoteCacheCKPTOp.

RecvRemoteCacheCKPTOp::RecvRemoteCacheCKPTOp(OpKernelConstruction* context)
  : OpKernel(context) {}

RecvRemoteCacheCKPTOp::~RecvRemoteCacheCKPTOp() {}

void RecvRemoteCacheCKPTOp::Compute(OpKernelContext* context) {
  auto rm = context->resource_manager();
  auto ndef = def();

  ContainerInfo cinfo;
  OP_REQUIRES_OK(context, cinfo.Init(rm, ndef, true /* use name */));

  RemoteCacheCKPTManager* ckpt_mgr = nullptr;
  OP_REQUIRES_OK(context, rm->LookupOrCreate<RemoteCacheCKPTManager>(
                              cinfo.container(), cinfo.name(), &ckpt_mgr,
                              [](RemoteCacheCKPTManager** p) -> Status {
                                *p = new RemoteCacheCKPTManager();
                                return Status::OK();
                              }));
  core::ScopedUnref scope(ckpt_mgr);

  const Tensor& cache_ckpt_key_tensor = context->input(0);
  const Tensor& cache_ckpt_tensor = context->input(1);

  OP_REQUIRES(context,
              TensorShapeUtils::IsScalar(cache_ckpt_key_tensor.shape()),
              errors::InvalidArgument("cache_ckpt_key is not a scalar: ",
                                      cache_ckpt_key_tensor.shape()
                                                           .DebugString()));
  OP_REQUIRES(context, TensorShapeUtils::IsScalar(cache_ckpt_tensor.shape()),
              errors::InvalidArgument("cache_ckpt is not a scalar: ",
                                      cache_ckpt_tensor.shape().DebugString()));

  const std::string& cache_ckpt_key = \
    cache_ckpt_key_tensor.scalar<tstring>()();
  ckpt_mgr->UpdateCacheCKPT(cache_ckpt_key, cache_ckpt_tensor);
}

REGISTER_KERNEL_BUILDER(Name("RecvRemoteCacheCKPT").Device(DEVICE_CPU),
                        RecvRemoteCacheCKPTOp);
} // End of namespace tensorflow
