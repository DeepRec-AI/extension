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

#include "deeprec_master/include/model_ready/grpc_model_ready_mgr.h"
#include "deeprec_master/include/rpc/rpc_manager.h"
#include "deeprec_master/include/status.h"
#include "deeprec_master/include/utils.h"
#include "deeprec_master/proto/model_ready/model_ready.pb.h"

namespace deeprecmaster {

GrpcModelReadyMgr::GrpcModelReadyMgr(SharedGrpcChannelPtr channel)
  : stub_(ModelReadyMgrService::NewStub(channel)) {}

Status GrpcModelReadyMgr::SetState(const std::string& task_name,
                                   const int32_t task_index, bool ready_state) {

  ModelReadyMgrSetStateRequest req;
  ModelReadyMgrSetStateResponse resp;

  req.set_task_name(task_name);
  req.set_task_index(task_index);
  req.set_ready_state(ready_state);

  grpc::ClientContext ctx;
  grpc::Status s = stub_->SetState(&ctx, req, &resp);
  if (!s.ok()) {
    return Grpc2Status(s);
  }
  if (resp.code() != 0) {
    Status st(deeprecmaster::error::Code(resp.code()), resp.msg());
    return st;
  }

  return Status::OK();
}

Status GrpcModelReadyMgr::GetState(const std::string& task_name,
                                   const int32_t task_index,
                                   bool* ready_state) {
  *ready_state = false;
  ModelReadyMgrGetStateRequest req;
  ModelReadyMgrGetStateResponse resp;

  req.set_task_name(task_name);
  req.set_task_index(task_index);

  grpc::ClientContext ctx;
  grpc::Status s = stub_->GetState(&ctx, req, &resp);
  if (!s.ok()) {
    return Grpc2Status(s);
  }
  if (resp.code() != 0) {
    *ready_state = false;
    Status st(deeprecmaster::error::Code(resp.code()), resp.msg());
    return st;
  }

  *ready_state = resp.ready_state();
  return Status::OK();
}


typedef std::function<Status(GrpcModelReadyMgr*)> ModelReadyMgrGrpcStubFn;

Status SetStateModelReadyMgr(const std::string& master_addr,
                             const std::string& task_name,
                             const int32_t task_index, bool ready_state) {
  ModelReadyMgrGrpcStubFn stub_fn = \
    std::bind(&GrpcModelReadyMgr::SetState, std::placeholders::_1, task_name,
              task_index, ready_state);

  return RPCMgr::RunStubWithRetry(master_addr, "SetState", stub_fn);
}

ModelReadyMgrQueryResult GetStateModelReadyMgr(const std::string& master_addr,
                                               const std::string& task_name,
                                               const int32_t task_index) {
  bool ready_state = false;
  ModelReadyMgrGrpcStubFn stub_fn = \
    std::bind(&GrpcModelReadyMgr::GetState, std::placeholders::_1, task_name,
              task_index, &ready_state);

  Status st = RPCMgr::RunStubWithRetry(master_addr, "GetState", stub_fn);

  struct ModelReadyMgrQueryResult result;
  result.status = st;
  result.ready_state = ready_state;

  return result;
}

} // End of namespace deeprecmaster
