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

#include "tensorflow/core/platform/env.h"

#include "tf_fault_tolerance/cc/common/cache_ckpt/cache_ckpt_storage.h"
#include "tf_fault_tolerance/cc/utils/cache_ckpt/naming.h"
#include "tf_fault_tolerance/cc/utils/platform/file.h"

namespace tensorflow {

//------------------------------------------------------------------------------
// functions of CacheCKPTStorage.
CacheCKPTStorage::CacheCKPTStorage(const int64 global_step)
  : global_step_(global_step) {};

CacheCKPTStorage::~CacheCKPTStorage() {};

int64 CacheCKPTStorage::global_step() const {
  return global_step_;
}

//------------------------------------------------------------------------------
// functions of MemoryCacheCKPTStorage.

MemoryCacheCKPTStorage::MemoryCacheCKPTStorage(const int64 global_step)
  : CacheCKPTStorage(global_step) {
  AllocationAttributes attr;
  attr.allocation_will_be_logged = true;

  Tensor meta_tmp(cpu_allocator(), DT_STRING, TensorShape({}), attr);
  cache_ckpt_meta_tensor_ = meta_tmp;

  Tensor data_tmp(cpu_allocator(), DT_STRING, TensorShape({}), attr);
  cache_ckpt_data_tensor_ = data_tmp;
}

MemoryCacheCKPTStorage::~MemoryCacheCKPTStorage() {}

Status MemoryCacheCKPTStorage::Write(const Tensor& ckpt_meta_tensor,
                                     const Tensor& ckpt_data_tensor) {
  cache_ckpt_meta_tensor_ = ckpt_meta_tensor;
  cache_ckpt_data_tensor_ = ckpt_data_tensor;
  return Status::OK();
}

Status MemoryCacheCKPTStorage::Write(const std::string& ckpt_meta_file_path,
                                     const std::string& ckpt_data_file_path,
                                     const bool delete_src_file) {
  Env* env = Env::Default();
  auto ckpt_meta_buffer = cache_ckpt_meta_tensor_.scalar<tstring>().data();
  TF_RETURN_IF_ERROR(
    ReadFileToString(env, ckpt_meta_file_path, ckpt_meta_buffer));

  auto ckpt_data_buffer = cache_ckpt_data_tensor_.scalar<tstring>().data();
  TF_RETURN_IF_ERROR(
    ReadFileToString(env, ckpt_data_file_path, ckpt_data_buffer));

  if (delete_src_file) {
    env->DeleteFile(ckpt_meta_file_path);
    env->DeleteFile(ckpt_data_file_path);
  }

  return Status::OK();
}

Status MemoryCacheCKPTStorage::Read(Tensor& ckpt_meta_tensor,
                                    Tensor& ckpt_data_tensor) const {
  ckpt_meta_tensor = cache_ckpt_meta_tensor_;
  ckpt_data_tensor = cache_ckpt_data_tensor_;
  return Status::OK();
}

Status MemoryCacheCKPTStorage::Read(std::string& ckpt_meta_file_path,
                                    std::string& ckpt_data_file_path,
                                    const bool get_ckpt_full_path) const {

  return errors::Unimplemented("MemoryCacheCKPTStorage::Read() is not implemented");
}

//------------------------------------------------------------------------------
// functions of POSIXFileCacheCKPTStorage.
POSIXFileCacheCKPTStorage::POSIXFileCacheCKPTStorage(
                             const std::string& cache_path,
                             const std::string& ckpt_filename_prefix,
                             const int64 global_step, const int32 shard,
                             const int32 num_shards, const bool is_local_ckpt,
                             const bool is_merged_meta)
  : CacheCKPTStorage(global_step), cache_path_(cache_path),
    is_local_ckpt_(is_local_ckpt) {
  cache_ckpt_file_path_prefix_ = \
    CacheCKPTFilePathPrefix(cache_path, ckpt_filename_prefix, global_step);
  if (is_local_ckpt && is_merged_meta) {
    cache_ckpt_meta_file_path_ = CKPTMetaFilename(cache_ckpt_file_path_prefix_);
  } else {
    cache_ckpt_meta_file_path_ = \
      CacheCKPTMetaFilename(cache_ckpt_file_path_prefix_, shard, num_shards);
  }
  cache_ckpt_data_file_path_ = \
    CKPTDataFilename(cache_ckpt_file_path_prefix_, shard, num_shards);
}

POSIXFileCacheCKPTStorage::~POSIXFileCacheCKPTStorage() {
  Env* env = Env::Default();
  Status s = env->DeleteFile(cache_ckpt_meta_file_path_);
  if (!s.ok()) {
    LOG(WARNING) << "POSIXFileCacheCKPTStorage: Failed to delete cache ckpt ("
                 << s.error_message() << ").";
  }
  s = env->DeleteFile(cache_ckpt_data_file_path_);
  if (!s.ok()) {
    LOG(WARNING) << "POSIXFileCacheCKPTStorage: Failed to delete cache ckpt ("
                 << s.error_message() << ").";
  }
}

Status POSIXFileCacheCKPTStorage::Write(const Tensor& ckpt_meta_tensor,
                                        const Tensor& ckpt_data_tensor) {
  Env* env = Env::Default();
  // Try to create cache_ckpt_path directory.
  TF_RETURN_IF_ERROR(env->RecursivelyCreateDir(cache_path_));

  // Write new cache ckpt to file.
  TF_RETURN_IF_ERROR(WriteStringToFile(env, cache_ckpt_meta_file_path_,
                       ckpt_meta_tensor.scalar<tstring>()()));

  TF_RETURN_IF_ERROR(WriteStringToFile(env, cache_ckpt_data_file_path_,
                       ckpt_data_tensor.scalar<tstring>()()));
  return Status::OK();
}

Status POSIXFileCacheCKPTStorage::Write(
                                    const std::string& ckpt_meta_file_path,
                                    const std::string& ckpt_data_file_path,
                                    const bool delete_src_file) {
  Env* env = Env::Default();
  // Try to create cache_ckpt_path directory.
  TF_RETURN_IF_ERROR(env->RecursivelyCreateDir(cache_path_));

  // move ckpt file to cache_ckpt_path directory.
  if (delete_src_file) {
    TF_RETURN_IF_ERROR(env->RenameFile(ckpt_meta_file_path,
                                       cache_ckpt_meta_file_path_));
    TF_RETURN_IF_ERROR(env->RenameFile(ckpt_data_file_path,
                                       cache_ckpt_data_file_path_));
  } else {
    TF_RETURN_IF_ERROR(io_utils::CopyFile(ckpt_meta_file_path,
                                          cache_ckpt_meta_file_path_));
    TF_RETURN_IF_ERROR(io_utils::CopyFile(ckpt_data_file_path,
                                          cache_ckpt_data_file_path_));
  }

  return Status::OK();
}

Status POSIXFileCacheCKPTStorage::Read(Tensor& ckpt_meta_tensor,
                                    Tensor& ckpt_data_tensor) const {
  Env* env = Env::Default();
  auto ckpt_meta_buffer = ckpt_meta_tensor.scalar<tstring>().data();
  TF_RETURN_IF_ERROR(ReadFileToString(env, cache_ckpt_meta_file_path_,
                                      ckpt_meta_buffer));
  auto ckpt_data_buffer = ckpt_data_tensor.scalar<tstring>().data();
  TF_RETURN_IF_ERROR(ReadFileToString(env, cache_ckpt_data_file_path_,
                                      ckpt_data_buffer));

  return Status::OK();
}

Status POSIXFileCacheCKPTStorage::Read(std::string& ckpt_meta_file_path,
                                       std::string& ckpt_data_file_path,
                                       const bool get_ckpt_full_path) const {
  if (get_ckpt_full_path) {
    ckpt_meta_file_path = cache_ckpt_meta_file_path_;
    ckpt_data_file_path = cache_ckpt_data_file_path_;
    return Status::OK();
  }

  ckpt_meta_file_path = cache_ckpt_file_path_prefix_;
  ckpt_data_file_path = cache_ckpt_file_path_prefix_;

  return Status::OK();
}

} // End of namespace tensorflow
