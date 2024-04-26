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

#include <mutex>

#include "deeprec_master/include/status.h"
#include "deeprec_master/proto/model_ready/model_ready.pb.h"

namespace deeprecmaster {

class ModelReadyManager {
 public:
  ModelReadyManager();
  ~ModelReadyManager();

  Status SetState(const std::string& task_name,
                  const int32_t task_index,
                  bool ready_state);

  Status GetState(const std::string& task_name,
                  const int32_t task_index,
                  bool* ready_state);

  static ModelReadyManager* GetInstance();

 private:
  // Variables.
  bool ready_state_;
  mutable std::mutex mu_;
};

} // End of namespace deeprecmaster
