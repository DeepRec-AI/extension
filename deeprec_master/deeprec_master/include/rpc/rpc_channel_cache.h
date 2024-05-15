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
#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "grpcpp/grpcpp.h"
#include "deeprec_master/include/status.h"

namespace deeprecmaster {

typedef std::shared_ptr<::grpc::Channel> SharedGrpcChannelPtr;

class GrpcChannelCache {
 public:
  GrpcChannelCache() {}
  ~GrpcChannelCache() {}
  SharedGrpcChannelPtr FindChannel(const std::string& target) const;
  Status UpdateChannel(
      const std::string& target,
      SharedGrpcChannelPtr channel,
      bool force=true);
  int64_t Size() const;

 private:
  mutable std::mutex mu_;
  std::unordered_map<std::string, SharedGrpcChannelPtr> channels_;
};

} // End of namespace deeprecmaster
