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

#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "deeprec_master/include/ps_resource_analyzer.h"
#include "deeprec_master/include/status.h"
#include "deeprec_master/proto/elastic_training.pb.h"

namespace deeprecmaster {

class SchedulerService {
 public:
  virtual ~SchedulerService() {}
  virtual std::string Start() = 0;
  virtual void Join() = 0;
};

SchedulerService* NewSchedulerService(const std::string& ip, int port);

}  // namespace deeprecmaster
