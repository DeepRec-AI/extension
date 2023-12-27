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

#include <utility>
#include <vector>

#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/lib/strings/numbers.h"

#include "tf_fault_tolerance/cc/common/cache_ckpt/cache_ckpt_manager.h"
#include "tf_fault_tolerance/cc/utils/cache_ckpt/naming.h"
#include "tf_fault_tolerance/cc/utils/cache_ckpt/storage_type.h"

namespace tensorflow {

//------------------------------------------------------------------------------
// functions of CacheCKPTManager.

/* static */ CacheCKPTManager* CacheCKPTManager::GetInstance() {
  static CacheCKPTManager*  instance = new CacheCKPTManager();
  return instance;
}

CacheCKPTManager::CacheCKPTManager() {
  local_storage_type_ = GetLocalCacheCKPTStorageTypeFromEnvVar();
  remote_storage_type_ = GetRemoteCacheCKPTStorageTypeFromEnvVar();

  LOG(INFO) << "CacheCKPTManager: store local cache CKPT in "
            << local_storage_type_ << ", and store remote cache ckpt in "
            << remote_storage_type_;
}

CacheCKPTManager::~CacheCKPTManager() {
  cache_ckpts_.clear();
}

Status CacheCKPTManager::GetOrCreateStorage(
                           const std::string& shard_key, int64 global_step,
                           const struct CacheCKPTManagerParams& params,
                           std::unique_ptr<CacheCKPTStorage>& storage_ptr) {
  bool already_exist = false;
  {
    tf_shared_lock l(mu_);
    already_exist = \
      cache_ckpts_.find(shard_key) != cache_ckpts_.end() && \
      (cache_ckpts_[shard_key])->global_step() == global_step;
  }

  if (!already_exist) {
    return CreateStorage(params, global_step, shard_key, storage_ptr);
  }

  // reuse.
  mutex_lock l(mu_);
  storage_ptr = std::move(cache_ckpts_[shard_key]);
  cache_ckpts_.erase(shard_key);

  return Status::OK();
}

void CacheCKPTManager::UpdateCacheCKPT(
                         const std::string& ckpt_key,
                         const Tensor& ckpt_meta_tensor,
                         const Tensor& ckpt_data_tensor,
                         const struct CacheCKPTManagerParams& params) {
  std::string shard_key;
  int64 global_step = 0;
  ParseCKPTKey(ckpt_key, shard_key, global_step);

  Status s;
  std::unique_ptr<CacheCKPTStorage> storage_ptr;
  s = GetOrCreateStorage(shard_key, global_step, params, storage_ptr);
  if (!s.ok()) {
    LOG(WARNING) << "CacheCKPTManager: Failed to get storage ("
                 << s.error_message() << ").";
  }

  s = storage_ptr->Write(ckpt_meta_tensor, ckpt_data_tensor);
  if (!s.ok()) {
    LOG(WARNING) << "CacheCKPTManager: Failed to update cache ckpt ("
                 << s.error_message() << ").";
    return;
  }

  mutex_lock l(mu_);
  cache_ckpts_[shard_key] = std::move(storage_ptr);
}

void CacheCKPTManager::UpdateCacheCKPT(
                         const std::string& ckpt_key,
                         const std::string& meta_file_path,
                         const std::string& data_file_path,
                         const bool delete_src_file,
                         const struct CacheCKPTManagerParams& params) {
  std::string shard_key;
  int64 global_step = 0;
  ParseCKPTKey(ckpt_key, shard_key, global_step);

  Status s;
  std::unique_ptr<CacheCKPTStorage> storage_ptr;
  s = GetOrCreateStorage(shard_key, global_step, params, storage_ptr);
  if (!s.ok()) {
    LOG(WARNING) << "CacheCKPTManager: Failed to get storage ("
                 << s.error_message() << ").";
  }

  s = storage_ptr->Write(meta_file_path, data_file_path, delete_src_file);
  if (!s.ok()) {
    LOG(WARNING) << "CacheCKPTManager: Failed to update cache ckpt ("
                 << s.error_message() << ").";
    return;
  }

  mutex_lock l(mu_);
  cache_ckpts_[shard_key] = std::move(storage_ptr);
}

bool CacheCKPTManager::TryToGetCacheCKPT(const std::string& ckpt_key,
                                         Tensor& ckpt_meta_tensor,
                                         Tensor& ckpt_data_tensor) {
  std::string shard_key;
  int64 global_step = 0;
  ParseCKPTKey(ckpt_key, shard_key, global_step);

  tf_shared_lock l(mu_);
  if (cache_ckpts_.count(shard_key) == 0 ||
      (cache_ckpts_[shard_key])->global_step() != global_step) {
    return false;
  }

  Status s = \
    (cache_ckpts_[shard_key])->Read(ckpt_meta_tensor,ckpt_data_tensor);
  if (!s.ok()) {
    LOG(WARNING) << "CacheCKPTManager: Failed get cache ckpt ("
                 << s.error_message() << ").";
    return false;
  }
  return true;
}

bool CacheCKPTManager::TryToGetCacheCKPT(const std::string& ckpt_key,
                                         const bool get_ckpt_full_path,
                                         std::string& ckpt_meta_file_path,
                                         std::string& ckpt_data_file_path) {
  std::string shard_key;
  int64 global_step = 0;
  ParseCKPTKey(ckpt_key, shard_key, global_step);

  tf_shared_lock l(mu_);
  if (cache_ckpts_.count(shard_key) == 0 || \
      (cache_ckpts_[shard_key])->global_step() != global_step) {
    return false;
  }

  Status s = \
    (cache_ckpts_[shard_key])->Read(ckpt_meta_file_path, ckpt_data_file_path,
                                    get_ckpt_full_path);
  if (!s.ok()) {
    LOG(WARNING) << "CacheCKPTManager: Failed get cache ckpt ("
                 << s.error_message() << ").";
    return false;
  }

  return true;
}

Status CacheCKPTManager::CreateStorage(
                           const struct CacheCKPTManagerParams& params,
                           const int64 global_step,
                           const std::string& shard_key,
                           std::unique_ptr<CacheCKPTStorage>& ptr) {
  bool is_local_ckpt = params.is_local_ckpt;
  const std::string& storage_type = params.storage_type;

  if (storage_type == StorageType::kMemoryType) {
    ptr.reset(new MemoryCacheCKPTStorage(global_step));
    return Status::OK();
  } else if (storage_type == StorageType::kPosixFileType) {
    const std::string& cache_path = params.cache_path;
    const std::string& filename_prefix = params.ckpt_filename_prefix;
    int32 shard, num_shards;
    ParseShardKey(shard_key, shard, num_shards);
    const bool is_merged_meta = params.is_merged_meta;
    ptr.reset(new POSIXFileCacheCKPTStorage(
                    cache_path, filename_prefix, global_step, shard, num_shards,
                    is_local_ckpt, is_merged_meta));
    return Status::OK();
  }

  return errors::InvalidArgument("CacheCKPTManager: Invalid Storage Type: "
                                 + storage_type);
}

std::string CacheCKPTManager::DebugString() const {
  std::string val = "CacheCKPTManager(cache_ckpt=[";
  for (auto iter = cache_ckpts_.begin(); iter != cache_ckpts_.end(); iter++) {
    int64 global_step = iter->second->global_step();
    val += "{global_step: " + std::to_string(global_step) + ", ";
    val += "shard_key: " + iter->first + "}";
  }
  val += "])";
  return val;
}

} // End of namespace tensorflow
