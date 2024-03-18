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
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "deeprec_master/include/status.h"

namespace deeprecmaster {

struct MonitorAction {
  int64_t timestamp;
  std::string task_name;
  int32_t task_index;
  std::string options;  // json format string
};

typedef std::pair<int64_t, std::string> TimeAndMetricsPair;

class MetricsMgr {
 public:
  Status AddNewMetric(const MonitorAction& actions);
  std::unordered_map<std::string, std::vector<TimeAndMetricsPair>>
  GetAllActions();
  Status GetTaskActions(const std::string& task_id,
                        std::vector<TimeAndMetricsPair>& actions);
  Status ClearAllActions();

  static MetricsMgr& GetInstance();

 private:
  MetricsMgr() : actions_count_(0){};

 private:
  std::set<std::string> tasks_;
  std::unordered_map<std::string, std::vector<TimeAndMetricsPair>>
      task_metrics_;
  uint64_t actions_count_;
};

}  // namespace deeprecmaster
