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

#ifndef TF_FAULT_TOLERANCE_CC_COMMON_CACHE_CKPT_CACHE_CKPT_MANAGER_H_
#define TF_FAULT_TOLERANCE_CC_COMMON_CACHE_CKPT_CACHE_CKPT_MANAGER_H_

#include <string>
#include <unordered_map>
#include <utility>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mutex.h"

#include "tf_fault_tolerance/cc/common/cache_ckpt/cache_ckpt_storage.h"

namespace tensorflow {

struct CacheCKPTManagerParams {
  bool is_local_ckpt; // input ckpt belongs to local ps or not
  bool is_merged_meta; // meta file is merged or not
  std::string cache_path; // path to cache dir
  std::string ckpt_filename_prefix; // without global step, shard and num_shards
  std::string storage_type; // storage type to store cache ckpt
};

class CacheCKPTManager {
 public:
  static CacheCKPTManager* GetInstance();

  void UpdateCacheCKPT(const std::string& ckpt_key,
                       const Tensor& ckpt_meta_tensor,
                       const Tensor& ckpt_data_tensor,
                       const struct CacheCKPTManagerParams& params);

  void UpdateCacheCKPT(const std::string& ckpt_key,
                       const std::string& ckpt_meta_file_path,
                       const std::string& ckpt_data_file_path,
                       const bool delete_src_file,
                       const struct CacheCKPTManagerParams& params);

  bool TryToGetCacheCKPT(const std::string& ckpt_key, Tensor& ckpt_meta_tensor,
                         Tensor& ckpt_data_tensor);

  bool TryToGetCacheCKPT(const std::string& ckpt_key,
                         const bool get_ckpt_full_path,
                         std::string& ckpt_meta_file_path,
                         std::string& ckpt_data_file_path);

  std::string DebugString() const;

 private:
  // Functions
  CacheCKPTManager();
  ~CacheCKPTManager();

  Status GetOrCreateStorage(const std::string& shard_key, int64 global_step,
                          const struct CacheCKPTManagerParams& params,
                          std::unique_ptr<CacheCKPTStorage>& storage_ptr);

  Status CreateStorage(const struct CacheCKPTManagerParams& params,
                       const int64 global_step, const std::string& shard_key,
                       std::unique_ptr<CacheCKPTStorage>& ptr);

  // Variables.
  mutex mu_;
  // format: key: <shard_id-num_shards>, value: <global_step, ckpt>
  std::unordered_map<std::string, std::unique_ptr<CacheCKPTStorage>>
    cache_ckpts_ GUARDED_BY(mu_);

  std::string local_storage_type_;
  std::string remote_storage_type_;
};

} // End of namespace tensorflow

#endif // End of TF_FAULT_TOLERANCE_CC_COMMON_CACHE_CKPT_CACHE_CKPT_MANAGER_H_
