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

#include "tf_fault_tolerance/cc/common/cache_ckpt/remote_cache_ckpt_manager.h"

namespace tensorflow {

//------------------------------------------------------------------------------
// functions of RemoteCacheCKPTManager.

RemoteCacheCKPTManager::RemoteCacheCKPTManager() {}

RemoteCacheCKPTManager::~RemoteCacheCKPTManager() {
  cache_ckpts_.clear();
}

void RemoteCacheCKPTManager::UpdateCacheCKPT(const std::string& ckpt_key,
                                             const Tensor& ckpt_tensor) {
  std::string shard_key;
  int64 global_step = 0;
  ParseCacheCKPTKey(ckpt_key, shard_key, global_step);

  cache_ckpts_[shard_key] = std::make_pair(global_step, ckpt_tensor);
}

bool RemoteCacheCKPTManager::TryGetCacheCKPT(const std::string& ckpt_key,
                                             Tensor* ckpt_tensor) {
  std::string shard_key;
  int64 global_step = 0;
  ParseCacheCKPTKey(ckpt_key, shard_key, global_step);

  if (cache_ckpts_.count(shard_key) != 0 && \
      cache_ckpts_[shard_key].first == global_step) {
    *ckpt_tensor = cache_ckpts_[shard_key].second;
    return true;
  }

  return false;
}

void RemoteCacheCKPTManager::ParseCacheCKPTKey(const std::string& ckpt_key,
                                               std::string& shard_key,
                                               int64& global_step) {
  // ckpt_key format: global_step:shard_id-num_shards
  std::vector<std::string> ckpt_key_vec = absl::StrSplit(ckpt_key, ":");
  CHECK_EQ(ckpt_key_vec.size(), 2);

  shard_key = ckpt_key_vec.back();
  CHECK(strings::safe_strto64(ckpt_key_vec.front(), &global_step));
}

std::string RemoteCacheCKPTManager::DebugString() const {
    std::string val = "RemoteCacheCKPTManager(cache_ckpt={";
    for (auto iter = cache_ckpts_.begin(); iter != cache_ckpts_.end(); iter++) {
      val += " " + std::to_string(iter->second.first) + "-" + iter->first + ",";
    }
    val += "})";

    return val;
}

} // End of namespace tensorflow
