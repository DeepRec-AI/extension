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
#include <unistd.h>
#include <chrono>
#include <cmath>

#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/numbers.h"

#include "deeprec_master/include/errors.h"
#include "deeprec_master/include/utils.h"

namespace deeprecmaster {

constexpr char kStreamRemovedMessage[] = "Stream removed";
static bool IsStreamRemovedError(const ::grpc::Status& s) {
  return !s.ok() && s.error_code() == ::grpc::StatusCode::UNKNOWN &&
         s.error_message() == kStreamRemovedMessage;
}

Status Grpc2Status(grpc::Status s) {
  if (s.ok()) {
    return Status::OK();
  } else {
    if (IsStreamRemovedError(s)) {
      return Status(deeprecmaster::error::UNAVAILABLE, s.error_message());
    }
    return Status(static_cast<deeprecmaster::error::Code>(s.error_code()),
                  s.error_message());
  }
}

int64_t GetTimeStamp(int64_t latency) {
  // Return Microseconds timestamp
  usleep(latency);
  std::chrono::time_point<std::chrono::system_clock, std::chrono::microseconds>
      tp = std::chrono::time_point_cast<std::chrono::microseconds>(
          std::chrono::system_clock::now());
  auto epoch = std::chrono::duration_cast<std::chrono::microseconds>(
      tp.time_since_epoch());
  return epoch.count();
}

int64_t GetBackOffTimeInUs(int retry_count) {
  static const int64_t initial_backoff = 2 * 1000000;  // 2s
  static const int64_t max_backoff = 60 * 1000000;     // 60s
  retry_count = std::min(retry_count, 8);
  int64_t pow = std::exp2(retry_count);
  return std::min(max_backoff, initial_backoff * pow);
}

namespace env_var {

Status ReadBoolFromEnvVar(absl::string_view env_var_name, bool default_val,
                          bool* value) {
  *value = default_val;
  const char* env_var_val = getenv(std::string(env_var_name).c_str());
  if (env_var_val == nullptr) {
    return Status::OK();
  }
  std::string str_value = absl::AsciiStrToLower(env_var_val);
  if (str_value == "0" || str_value == "false") {
    *value = false;
    return Status::OK();
  } else if (str_value == "1" || str_value == "true") {
    *value = true;
    return Status::OK();
  }
  return error::InvalidArgument(absl::StrCat(
      "Failed to parse the env-var ${", env_var_name, "} into bool: ",
      env_var_val, ". Use the default value: ", default_val));
}

Status ReadInt64FromEnvVar(absl::string_view env_var_name, int64_t default_val,
                           int64_t* value) {
  *value = default_val;
  const char* env_var_val = getenv(std::string(env_var_name).c_str());
  if (env_var_val == nullptr) {
    return Status::OK();
  }
  if (absl::SimpleAtoi(env_var_val, value)) {
    return Status::OK();
  }
  return error::InvalidArgument(absl::StrCat(
      "Failed to parse the env-var ${", env_var_name, "} into int64: ",
      env_var_val, ". Use the default value: ", default_val));
}

Status ReadStringFromEnvVar(absl::string_view env_var_name, absl::string_view default_val,
                            std::string* value) {
  const char* env_var_val = getenv(std::string(env_var_name).c_str());
  if (env_var_val != nullptr) {
    *value = env_var_val;
  } else {
    *value = std::string(default_val);
  }
  return Status::OK();
}

} // End of namespace env_var

}  // namespace deeprecmaster
