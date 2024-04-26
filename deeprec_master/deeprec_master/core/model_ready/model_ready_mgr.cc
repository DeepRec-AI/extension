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

#include "deeprec_master/include/model_ready/model_ready_mgr.h"
#include "deeprec_master/include/status.h"

namespace deeprecmaster {

ModelReadyManager::ModelReadyManager()
  : ready_state_(false) {}

ModelReadyManager::~ModelReadyManager() {}

ModelReadyManager* ModelReadyManager::GetInstance() {
  static ModelReadyManager* sg_manager_ptr = new ModelReadyManager();
  return sg_manager_ptr;
}

Status ModelReadyManager::SetState(const std::string& task_name,
                                   const int32_t task_index,
                                   bool ready_state) {
  std::lock_guard<std::mutex> lock(mu_);
  ready_state_ = ready_state;

  return Status::OK();
}

Status ModelReadyManager::GetState(const std::string& task_name,
                                   const int32_t task_index,
                                   bool* ready_state) {
  std::lock_guard<std::mutex> lock(mu_);
  *ready_state = ready_state_;

  return Status::OK();
}

} // End of namespace deeprecmaster
