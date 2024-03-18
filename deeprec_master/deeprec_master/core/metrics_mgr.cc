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

#include "deeprec_master/include/metrics_mgr.h"

#include <unistd.h>
#include <algorithm>
#include <random>
#include "grpcpp/grpcpp.h"

#include "deeprec_master/include/errors.h"
#include "deeprec_master/include/logging.h"
#include "deeprec_master/include/status.h"
#include "deeprec_master/include/utils.h"

namespace deeprecmaster {

namespace {
std::string GetTaskId(const MonitorAction& actions) {
  return actions.task_name + "#" + std::to_string(actions.task_index);
}

std::string GetTaskId(const std::string& task_name, int32_t task_index) {
  return task_name + "#" + std::to_string(task_index);
}

}  // namespace

Status MetricsMgr::AddNewMetric(const MonitorAction& actions) {
  std::string task_id = GetTaskId(actions);

  if (tasks_.find(task_id) == tasks_.end()) {
    tasks_.insert(task_id);
  }
  if (task_metrics_.find(task_id) != task_metrics_.end()) {
    task_metrics_[task_id].emplace_back(
        TimeAndMetricsPair(actions.timestamp, actions.options));
  } else {
    task_metrics_.insert({task_id, std::vector<TimeAndMetricsPair>{TimeAndMetricsPair(actions.timestamp, actions.options)}});
  }
  actions_count_++;
  return Status::OK();
}

std::unordered_map<std::string, std::vector<TimeAndMetricsPair>> MetricsMgr::GetAllActions() {
  std::unordered_map<std::string, std::vector<TimeAndMetricsPair>> actions;
  actions.reserve(task_metrics_.size());
  actions.swap(task_metrics_);
  return actions;
}

Status MetricsMgr::GetTaskActions(const std::string& task_id,
                                  std::vector<TimeAndMetricsPair>& actions) {
  actions.clear();
  if (tasks_.find(task_id) != tasks_.end()) {
    actions.swap(task_metrics_[task_id]);
  } else {
    std::string msg = "Not found actions for " + task_id;
    LogWarn("%s", msg);
    return error::NotFound(msg);
  }

  return Status::OK();
}

Status MetricsMgr::ClearAllActions() {
  for (auto& it : task_metrics_) {
    it.second.clear();
  }
  actions_count_ = 0;
  return Status::OK();
}

MetricsMgr& MetricsMgr::GetInstance() {
  static MetricsMgr metrics_mgr;
  return metrics_mgr;
}

}  // namespace deeprecmaster
