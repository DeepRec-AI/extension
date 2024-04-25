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

#include <functional>
#include <memory>
#include <string>

#include "grpcpp/grpcpp.h"

#include "deeprec_master/include/logging.h"
#include "deeprec_master/include/rpc/rpc_channel_cache.h"
#include "deeprec_master/include/status.h"
#include "deeprec_master/include/utils.h"

namespace deeprecmaster {

typedef std::shared_ptr<::grpc::Channel> SharedGrpcChannelPtr;

class RPCMgr {
 public:
  static RPCMgr* GetInstance();
  static Status NewHostPortGrpcChannel(const std::string& target,
                                       SharedGrpcChannelPtr* channel_ptr);
  static int64_t GetStubCallRetryCount();

  template <typename T>
  static Status RunStubWithRetry(const std::string& target,
                                 const std::string& stub_name,
                                 const std::function<Status(T*)>& stub_fn);

  Status FindChannel(const std::string& target,
                     SharedGrpcChannelPtr* channel_ptr);

 private:
  // Variables.
  GrpcChannelCache channel_cache_;

  //Functions.
  RPCMgr();
  ~RPCMgr();
};

RPCMgr* GetRPCMgr();

template <typename T>
Status RPCMgr::RunStubWithRetry(const std::string& target,
                                const std::string& stub_name,
                                const std::function<Status(T*)>& stub_fn) {
  Status st = Status::OK();
  int64_t max_retry_count = RPCMgr::GetStubCallRetryCount();
  for (int retry_count = 0; retry_count < max_retry_count; ++retry_count) {
    SharedGrpcChannelPtr channel;
    st = GetRPCMgr()->FindChannel(target, &channel);
    if (!channel || !st.ok()) {
      continue;
    }
    T stub(channel);
    st = stub_fn(&stub);
    if (st.code() == error::Code::UNAVAILABLE) {
      useconds_t sleep_usec = GetBackOffTimeInUs(retry_count);
      LogWarn("DeepRecMaster Service unavailable for %s rpc, wait %.1fs ...",
              stub_name, sleep_usec/1000000.0);
      usleep(sleep_usec);
    } else {
      break;
    }
  }
  return st;
}

} // End of namespace deeprecmaster
