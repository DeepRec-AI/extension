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

#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"

#include "deeprec_master/include/errors.h"
#include "deeprec_master/include/rpc/rpc_manager.h"

namespace deeprecmaster {

namespace {

Status ValidateHostPortPair(const std::string& host_port) {
  // Minimally ensure that the host_port is valid
  uint32_t port = 0;
  std::vector<std::string> parts = absl::StrSplit(host_port, ':');
  // Must be host:port, port must be a number, host must not contain a '/'.
  if (parts.size() != 2 || !absl::SimpleAtoi(parts[1], &port) ||
      parts[0].find("/") != std::string::npos) {
    return error::InvalidArgument("Could not interpret \"" + host_port +
                                   "\" as a host-port pair.");
  }
  return Status::OK();
}

} // End of namespace anonymous namespace

RPCMgr::RPCMgr() {}

RPCMgr::~RPCMgr() {}

RPCMgr* RPCMgr::GetInstance() {
  static RPCMgr* sg_rpc_mgr = new RPCMgr();
  return sg_rpc_mgr;
}

Status RPCMgr::NewHostPortGrpcChannel(const std::string& target,
                                      SharedGrpcChannelPtr* channel_ptr) {
  RETURN_IF_ERROR(ValidateHostPortPair(target));

  ::grpc::ChannelArguments args;
  args.SetInt(GRPC_ARG_MAX_MESSAGE_LENGTH, std::numeric_limits<int32_t>::max());
  args.SetInt(GRPC_ARG_MAX_RECONNECT_BACKOFF_MS, 1000);
  *channel_ptr = ::grpc::CreateCustomChannel(
      "dns:///" + target, ::grpc::InsecureChannelCredentials(), args);

  return Status::OK();
}

int64_t RPCMgr::GetStubCallRetryCount() {
  auto fn = [](const char* env_var_name, int64_t default_val)->int64_t {
      int64_t value = default_val;
      env_var::ReadInt64FromEnvVar(env_var_name, default_val, &value);
      return value;
  };
  static const int64_t value =
    fn("DEEPREC_MASTER_STUB_RPC_CALL_RETRY_COUNT", 7);
  return value;
}

Status RPCMgr::FindChannel(const std::string& target,
                           SharedGrpcChannelPtr* channel_ptr) {
  SharedGrpcChannelPtr channel = nullptr;
  channel = channel_cache_.FindChannel(target);
  if (channel) {
    *channel_ptr = channel;
    return Status::OK();
  }

  RETURN_IF_ERROR(NewHostPortGrpcChannel(target, &channel));
  channel_cache_.UpdateChannel(target, channel, false);

  *channel_ptr = channel;
  return Status::OK();
}

RPCMgr* GetRPCMgr() {
  return RPCMgr::GetInstance();
}

} // End of namespace deeprecmaster
