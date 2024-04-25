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
#include "deeprec_master/proto/data_distributor/data_manager.grpc.pb.h"

namespace deeprecmaster {

typedef std::shared_ptr<::grpc::Channel> SharedGrpcChannelPtr;

class GrpcDataManager {
 public:
  explicit GrpcDataManager(SharedGrpcChannelPtr channel);

  Status Init(const std::string& task_name, const int32_t task_index,
              const std::string& config);

  Status IsReady(const std::string& task_name, const int32_t task_index,
                 bool* is_ready);

  Status GetSlice(const std::string& task_name, const int32_t task_index,
                  std::string* slice_tag, std::string* slice_prefix,
                  int64_t* slice_size, std::vector<std::string>* slice_data);

  Status GetDataState(const std::string& task_name, const int32_t task_index,
                      std::string* data_state);

  Status StartDataDispatch(const std::string& task_name,
                           const int32_t task_index);

  Status StopDataDispatchAndGetDataState(const std::string& task_name,
                                         const int32_t task_index,
                                         std::string* data_state);

 private:
  std::unique_ptr<DataManagerService::Stub> stub_;
};

struct DataManagerQueryResult {
  Status status;
  bool is_ready;
  std::string slice_tag;
  std::string slice_prefix;
  int64_t slice_size;
  std::vector<std::string> slice_data;
  std::string data_state;
};

Status InitDataManager(const std::string& master_addr,
                       const std::string& task_name,
                       const int32_t task_index, const std::string& config);

DataManagerQueryResult IsReadyDataManager(const std::string& master_addr,
                                          const std::string& task_name,
                                          const int32_t task_index);

DataManagerQueryResult GetSliceFromDataManager(const std::string& master_addr,
                                               const std::string& task_name,
                                               const int32_t task_index);

DataManagerQueryResult GetDataStateFromDataManager(
    const std::string& master_addr, const std::string& task_name,
    const int32_t task_index);

Status DataManagerStartDataDispatch(const std::string& master_addr,
                                    const std::string& task_name,
                                    const int32_t task_index);

DataManagerQueryResult DataManagerStopDataDispatchAndGetDataState(
    const std::string& master_addr, const std::string& task_name,
    const int32_t task_index);

} // End of namespace deeprecmaster
