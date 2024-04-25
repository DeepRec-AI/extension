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

#include "absl/strings/escaping.h"

#include "deeprec_master/include/data_distributor/grpc_data_manager.h"
#include "deeprec_master/include/rpc/rpc_manager.h"
#include "deeprec_master/include/status.h"
#include "deeprec_master/include/utils.h"
#include "deeprec_master/proto/data_distributor/data_manager.pb.h"

namespace deeprecmaster {

//------------------------------------------------------------------------------
// Functions of class GrpcDataManager.
GrpcDataManager::GrpcDataManager(SharedGrpcChannelPtr channel)
  :stub_(DataManagerService::NewStub(channel)) {}

Status GrpcDataManager::Init(const std::string& task_name,
                             const int32_t task_index,
                             const std::string& config) {
  DataManagerInitRequest req;
  DataManagerCommonResponse resp;

  std::string config_base64;
  absl::Base64Escape(config, &config_base64);

  req.set_task_name(task_name);
  req.set_task_index(task_index);
  req.set_config(config_base64);

  grpc::ClientContext ctx;
  grpc::Status s = stub_->InitDataManager(&ctx, req, &resp);
  if (!s.ok()) {
    return Grpc2Status(s);
  }
  if (resp.code() != 0) {
    Status st(deeprecmaster::error::Code(resp.code()), resp.msg());
    return st;
  }

  return Status::OK();
}

Status GrpcDataManager::IsReady(const std::string& task_name,
                                const int32_t task_index, bool* is_ready) {
  *is_ready = false;
  DataManagerIsReadyRequest req;
  DataManagerIsReadyResponse resp;

  req.set_task_name(task_name);
  req.set_task_index(task_index);

  grpc::ClientContext ctx;
  grpc::Status s = stub_->IsReadyDataManager(&ctx, req, &resp);
  if (!s.ok()) {
    return Grpc2Status(s);
  }
  if (resp.code() != 0) {
    *is_ready = false;
    Status st(deeprecmaster::error::Code(resp.code()), resp.msg());
    return st;
  }

  *is_ready = resp.is_ready();
  return Status::OK();
}

Status GrpcDataManager::GetSlice(const std::string& task_name,
                                 const int32_t task_index,
                                 std::string* slice_tag,
                                 std::string* slice_prefix,
                                 int64_t* slice_size,
                                 std::vector<std::string>* slice_data) {
  slice_tag->clear();
  slice_prefix->clear();
  slice_data->clear();
  DataManagerGetSliceRequest req;
  DataManagerGetSliceResponse resp;

  req.set_task_name(task_name);
  req.set_task_index(task_index);

  grpc::ClientContext ctx;
  grpc::Status s = stub_->GetSliceFromDataManager(&ctx, req, &resp);
  if (!s.ok()) {
    return Grpc2Status(s);
  }

  if (resp.code() != 0) {
    Status st(deeprecmaster::error::Code(resp.code()), resp.msg());
    return st;
  }

  *slice_tag = resp.slice_tag();
  *slice_prefix = resp.slice_prefix();
  *slice_size = resp.slice_size();
  for (const auto& data: resp.slice_data()) {
    slice_data->push_back(data);
  }
  
  return Status::OK();
}

Status GrpcDataManager::GetDataState(const std::string& task_name,
                                     const int32_t task_index,
                                     std::string* data_state) {
  data_state->clear();

  DataManagerGetDataStateRequest req;
  DataManagerGetDataStateResponse resp;

  req.set_task_name(task_name);
  req.set_task_index(task_index);

  grpc::ClientContext ctx;
  grpc::Status s = stub_->GetDataStateFromDataManager(&ctx, req, &resp);
  if (!s.ok()) {
    return Grpc2Status(s);
  }
  if (resp.code() != 0) {
    Status st(deeprecmaster::error::Code(resp.code()), resp.msg());
    return st;
  }

  *data_state = resp.data_state();
  return Status::OK();
}

Status GrpcDataManager::StartDataDispatch(const std::string& task_name,
                                          const int32_t task_index) {
  DataManagerStartDataDispatchRequest req;
  DataManagerStartDataDispatchRespone resp;

  req.set_task_name(task_name);
  req.set_task_index(task_index);

  grpc::ClientContext ctx;
  grpc::Status s = stub_->DataManagerStartDataDispatch(&ctx, req, &resp);
  if (!s.ok()) {
    return Grpc2Status(s);
  }
  if (resp.code() != 0) {
    Status st(deeprecmaster::error::Code(resp.code()), resp.msg());
    return st;
  }

  return Status::OK();
}

Status GrpcDataManager::StopDataDispatchAndGetDataState(
         const std::string& task_name, const int32_t task_index,
         std::string* data_state) {
  data_state->clear();

  DataManagerStopDataDispatchAndGetDataStateRequest req;
  DataManagerStopDataDispatchAndGetDataStateResponse resp;

  req.set_task_name(task_name);
  req.set_task_index(task_index);

  grpc::ClientContext ctx;
  grpc::Status s = \
    stub_->DataManagerStopDataDispatchAndGetDataState(&ctx, req, &resp);
  if (!s.ok()) {
    return Grpc2Status(s);
  }
  if (resp.code() != 0) {
    Status st(deeprecmaster::error::Code(resp.code()), resp.msg());
    return st;
  }

  *data_state = resp.data_state();
  return Status::OK();
}

//------------------------------------------------------------------------------
// Functions.

typedef std::function<Status(GrpcDataManager*)> DataManagerGrpcStubFn;

Status InitDataManager(const std::string& master_addr,
                       const std::string& task_name,
                       const int32_t task_index, const std::string& config) {
  DataManagerGrpcStubFn stub_fn = \
    std::bind(&GrpcDataManager::Init, std::placeholders::_1, task_name,
              task_index, config);

  return RPCMgr::RunStubWithRetry(master_addr, "Init", stub_fn);
}

DataManagerQueryResult IsReadyDataManager(const std::string& master_addr,
                                          const std::string& task_name,
                                          const int32_t task_index) {
  bool is_ready = false;
  DataManagerGrpcStubFn stub_fn = \
    std::bind(&GrpcDataManager::IsReady, std::placeholders::_1, task_name,
              task_index, &is_ready);

  Status st = RPCMgr::RunStubWithRetry(master_addr, "IsReady", stub_fn);

  DataManagerQueryResult result;
  result.status = st;
  result.is_ready = is_ready;
  return result;
}

DataManagerQueryResult GetSliceFromDataManager(const std::string& master_addr,
                                               const std::string& task_name,
                                               const int32_t task_index) {
  std::string slice_tag;
  std::string slice_prefix;
  int64_t slice_size = 0;
  std::vector<std::string> slice_data;
  DataManagerGrpcStubFn stub_fn = \
    std::bind(&GrpcDataManager::GetSlice, std::placeholders::_1, task_name,
              task_index, &slice_tag, &slice_prefix, &slice_size, &slice_data);

  Status st =RPCMgr::RunStubWithRetry(master_addr, "GetSlice", stub_fn);
  DataManagerQueryResult result;
  result.status = st;
  result.is_ready = true;
  result.slice_tag = std::move(slice_tag);
  result.slice_prefix = std::move(slice_prefix);
  result.slice_size = slice_size;
  result.slice_data = std::move(slice_data);
  return result;
}

DataManagerQueryResult GetDataStateFromDataManager(
    const std::string& master_addr, const std::string& task_name,
    const int32_t task_index) {
  std::string data_state;
  DataManagerGrpcStubFn stub_fn = \
    std::bind(&GrpcDataManager::GetDataState, std::placeholders::_1, task_name,
              task_index, &data_state);

  Status st = \
    RPCMgr::RunStubWithRetry(master_addr, "GetDataState", stub_fn);
  DataManagerQueryResult result;
  result.status = st;
  result.data_state = std::move(data_state);

  return result;
}

Status DataManagerStartDataDispatch(const std::string& master_addr,
                                    const std::string& task_name,
                                    const int32_t task_index) {
  DataManagerGrpcStubFn stub_fn = \
    std::bind(&GrpcDataManager::StartDataDispatch, std::placeholders::_1,
              task_name, task_index);

  return RPCMgr::RunStubWithRetry(master_addr, "StartDataDispatch", stub_fn);
}

DataManagerQueryResult DataManagerStopDataDispatchAndGetDataState(
    const std::string& master_addr, const std::string& task_name,
    const int32_t task_index) {
  std::string data_state;
  DataManagerGrpcStubFn stub_fn = \
    std::bind(&GrpcDataManager::StopDataDispatchAndGetDataState,
              std::placeholders::_1, task_name, task_index, &data_state);

  Status st = \
    RPCMgr::RunStubWithRetry(master_addr,
                             "StopDataDispatchAndGetDataState", stub_fn);

  DataManagerQueryResult result;
  result.status = st;
  result.data_state = std::move(data_state);
  return result;
}

} // End of namespace deeprecmaster
