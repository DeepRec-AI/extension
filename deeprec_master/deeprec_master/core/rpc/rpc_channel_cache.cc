/* Copyright 2024 The DeepRec Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=============================================================================*/

#include "deeprec_master/include/rpc/rpc_channel_cache.h"

namespace deeprecmaster {

SharedGrpcChannelPtr GrpcChannelCache::FindChannel(
    const std::string& target) const {
  SharedGrpcChannelPtr ch = nullptr;
  {
    std::lock_guard<std::mutex> lock(mu_);
    const auto& it = channels_.find(target);
    if (it != channels_.end()) {
      ch = it->second;
    }
  }
  return ch;
}

Status GrpcChannelCache::UpdateChannel(
    const std::string& target, SharedGrpcChannelPtr channel, bool force) {
  SharedGrpcChannelPtr ch = channel;
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto ret = channels_.insert({target, ch});
    if (force && !ret.second) {
      ret.first->second = ch;
    }
  }
  return Status::OK();
}

int64_t GrpcChannelCache::Size() const {
  std::lock_guard<std::mutex> lock(mu_);
  return channels_.size();
}

} // End of namespace deeprecmaster
