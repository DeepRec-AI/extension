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

#ifndef TF_FAULT_TOLERANCE_CC_COMMON_CACHE_CKPT_REMOTE_CACHE_CKPT_MANAGER_H_
#define TF_FAULT_TOLERANCE_CC_COMMON_CACHE_CKPT_REMOTE_CACHE_CKPT_MANAGER_H_

#include <string>
#include <unordered_map>
#include <utility>

#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {

class RemoteCacheCKPTManager : public ResourceBase {
 public:
  RemoteCacheCKPTManager();

  ~RemoteCacheCKPTManager();

  void UpdateCacheCKPT(const std::string& ckpt_key, const Tensor& ckpt_tensor);

  bool TryGetCacheCKPT(const std::string& ckpt_key, Tensor* ckpt_tensor);

  std::string DebugString() const override;

 private:
  void ParseCacheCKPTKey(const std::string& ckpt_key, std::string& shard_key,
                         int64& global_step);

  // INFO: Thread Safety.
  // cache_ckpts_ is only written when saving checkpoint and only read when
  // restoring from checkpoint. so there will be no read-write competition.
  //
  // According to thread safety rule of container
  // (https://en.cppreference.com/w/cpp/container#Thread_safety). Different
  // elements in the same container can be modified concurrently by different
  // threads, except for the elements of std::vector<bool> (since C++11).
  // key: <shard_id-num_shards>, value: <global_step, ckpt>
  std::unordered_map<std::string, std::pair<int64, Tensor>> cache_ckpts_;
};

} // End of namespace tensorflow

#endif // End of TF_FAULT_TOLERANCE_CC_COMMON_CACHE_CKPT_REMOTE_CACHE_CKPT_MANAGER_H_
