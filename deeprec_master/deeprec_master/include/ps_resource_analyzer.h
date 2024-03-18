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

#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "deeprec_master/include/ps_resource.h"
#include "deeprec_master/include/status.h"
#include "deeprec_master/proto/elastic_training.pb.h"

namespace deeprecmaster {
class PsResourceAnalyzer {
 public:
  Status EstimatePSResource(bool hasev, int32_t psnum, int64_t psmem);
  PSResource GetEstimatedPSResource() const;
  void SetTFConfig(std::string);
  std::string GetTFConfig();
  void SetState(ElasticTrainingState s, int32_t ps_num, int32_t worker_num);
  ElasticTrainingState GetState();

  ScalingAction IsReadyScaling(int& ps_num);
  void ReadyToUpdate();

  static PsResourceAnalyzer* GetInstance() {
    if (psres_analyzer_ == nullptr) {
      psres_analyzer_ = new PsResourceAnalyzer();
    }
    return psres_analyzer_;
  };

 private:
  PsResourceAnalyzer()
      : tfconfig_(""),
        old_ps_num_(0),
        new_ps_num_(0),
        worker_num_(0),
        counter_(0) {}

  PsResourceAnalyzer(const PsResourceAnalyzer&) = delete;
  PsResourceAnalyzer& operator=(const PsResourceAnalyzer&) = delete;

 private:
  mutable std::mutex mu_;
  std::condition_variable cv_;
  static PsResourceAnalyzer* psres_analyzer_;
  PSResource estimated_ps_res_;
  std::string tfconfig_;

  ElasticTrainingState state_;
  int32_t old_ps_num_;
  int32_t new_ps_num_;
  int32_t worker_num_;
  int counter_;
};

}  // namespace deeprecmaster
