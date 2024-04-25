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

#include "deeprec_master/include/data_distributor/data_state.h"

namespace deeprecmaster {

DataState::DataState() {}

DataState::~DataState() {}

Status DataState::Init(const DataConfig& config) {
  data_config_.CopyFrom(config);
  return Status::OK();
}

void DataState::UpdateEpochIndex(int64_t value) {
  data_config_.mutable_options()->set_epoch_index(value);
}

void DataState::UpdateShuffleSeed(int64_t value) {
  data_config_.mutable_options()->set_shuffle_seed(value);
}

void DataState::UpdateSliceIndex(int64_t value) {
  data_config_.mutable_options()->set_slice_index(value);
}

bool DataState::SerializeToBase64String(std::string* output) {
  std::string data_state_str;
  if (data_config_.SerializeToString(&data_state_str)) {
    absl::Base64Escape(data_state_str, output);
    return true;
  };
  return false;
}

}; // End of namespace deeprecmaster
