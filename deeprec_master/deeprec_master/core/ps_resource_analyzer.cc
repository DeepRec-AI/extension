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
#include "deeprec_master/include/ps_resource_analyzer.h"

#include <unistd.h>
#include <chrono>
#include <map>
#include <thread>

#include "deeprec_master/include/errors.h"
#include "deeprec_master/include/logging.h"
#include "deeprec_master/include/utils.h"

namespace deeprecmaster {

Status PsResourceAnalyzer::EstimatePSResource(bool hasev, int32_t psnum,
                                              int64_t psmem) {
  std::lock_guard<std::mutex> lock(mu_);
  estimated_ps_res_.set_initialized();
  estimated_ps_res_.set_has_ev(hasev);
  estimated_ps_res_.set_ps_num(psnum);
  estimated_ps_res_.set_ps_memory(psmem);
  return Status::OK();
}

PSResource PsResourceAnalyzer::GetEstimatedPSResource() const {
  std::lock_guard<std::mutex> lock(mu_);
  return estimated_ps_res_;
}

void PsResourceAnalyzer::SetTFConfig(std::string tfconfig) {
  std::lock_guard<std::mutex> lock(mu_);
  tfconfig_ = tfconfig;
  return;
}

std::string PsResourceAnalyzer::GetTFConfig() {
  std::lock_guard<std::mutex> lock(mu_);
  return tfconfig_;
}

void PsResourceAnalyzer::SetState(ElasticTrainingState s, int32_t ps_num,
                                  int32_t worker_num) {
  std::lock_guard<std::mutex> lock(mu_);
  LogInfo("ElasticTrainingService SetState: %d", state_);
  state_ = s;
  worker_num_ = worker_num;
  if (SCALING == state_) {
    new_ps_num_ = ps_num;
  } else {
    old_ps_num_ = ps_num;
  }
}

ElasticTrainingState PsResourceAnalyzer::GetState() {
  std::lock_guard<std::mutex> lock(mu_);
  return state_;
}

ScalingAction PsResourceAnalyzer::IsReadyScaling(int& ps_num) {
  std::lock_guard<std::mutex> lock(mu_);
  ScalingAction scaling_action(NONE);
  if (SCALING == state_) {
    scaling_action = old_ps_num_ < new_ps_num_ ? SCALING_UP : SCALING_DOWN;
    ps_num = new_ps_num_;
  }
  return scaling_action;
}

void PsResourceAnalyzer::ReadyToUpdate() {
  {
    std::unique_lock<std::mutex> lock(mu_);
    ++counter_;  // 增加计数
    if (counter_ >= worker_num_) {
      state_ = All_SESSION_CLOSED;
      cv_.notify_all();
    } else {
      cv_.wait(lock, [this]() { return counter_ >= worker_num_; });
    }
  }
  
  LogInfo("ElasticTrainingService State: %d", state_);

  while (true) {
    usleep(1000);
    std::lock_guard<std::mutex> lock(mu_);
    if (READY == state_) {
      break;
    }
  }
  LogInfo("ElasticTrainingService State: %d", state_);
  return;
}

PsResourceAnalyzer* PsResourceAnalyzer::psres_analyzer_ = nullptr;

}  // namespace deeprecmaster
