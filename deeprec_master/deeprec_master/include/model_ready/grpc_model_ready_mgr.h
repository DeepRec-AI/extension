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

#include "grpcpp/grpcpp.h"

#include "deeprec_master/include/status.h"
#include "deeprec_master/proto/model_ready/model_ready.grpc.pb.h"

namespace deeprecmaster {

typedef std::shared_ptr<::grpc::Channel> SharedGrpcChannelPtr;

class GrpcModelReadyMgr {
 public:
  explicit GrpcModelReadyMgr(SharedGrpcChannelPtr channel);

  Status SetState(const std::string& task_name, const int32_t task_index,
                  bool ready_state);

  Status GetState(const std::string& task_name, const int32_t task_index,
                  bool* ready_state);

 private:
  std::unique_ptr<ModelReadyMgrService::Stub> stub_;
};

struct ModelReadyMgrQueryResult {
  Status status;
  bool ready_state;
};

Status SetStateModelReadyMgr(const std::string& master_addr,
                             const std::string& task_name,
                             const int32_t task_index, bool ready_state);

ModelReadyMgrQueryResult GetStateModelReadyMgr(const std::string& master_addr,
                                               const std::string& task_name,
                                               const int32_t task_index);

} // End of namespace deeprecmaster
