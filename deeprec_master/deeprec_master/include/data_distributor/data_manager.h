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

#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unistd.h>

#include "deeprec_master/include/data_distributor/data_slice_mgr.h"
#include "deeprec_master/include/data_distributor/data_state.h"
#include "deeprec_master/include/global_worker_queue.h"
#include "deeprec_master/include/status.h"
#include "deeprec_master/proto/data_distributor/data_config.pb.h"

namespace deeprecmaster {

class DataManagerConfig {
 public:
  DataManagerConfig();
  Status ParseDataConfigFromString(const std::string& config_pb);
  Status ValidateDataConfig();
  const DataConfig& GetDataConfig() const;

 private:
  Status ValidatePosixFileDataConfig();
  DataConfig data_config_;
};

class DataManager {
 public:
  DataManager();
  ~DataManager();

  Status Init(const std::string& task_name, const int32_t task_index,
              const std::string& config_pb);
  Status IsReady(const std::string& task_name, const int32_t task_index,
                 bool* is_ready);
  bool IsReady();
  Status GetSlice(const std::string& task_name, int32_t task_index,
                  std::string* slice_tag, std::string* slice_prefix,
                  int64_t* slice_size, std::vector<std::string>* slice_data);
  Status GetDataState(const std::string& task_name, const int32_t task_index,
                      std::string* data_state);
  Status StartDataDispatch(const std::string& task_name, const int32_t task_index);
  Status StopDataDispatch(const std::string& task_name, const int32_t task_index);

  static DataManager* GetInstance();

 private:
  // Functions.
  void Clear();
  Status DoInit();
  void UpdateDataState();
  Status DoInitFromFileMeta();
  Status DoInitFromFileDir();
  Status DoInitGlabalSliceQueue();

  // Variables.
  bool is_ready_;
  bool continue_dispatch_data_;
  uint64_t total_slice_count_;
  uint64_t processed_slice_count_;
  DataManagerConfig mgr_config_;
  DataSliceMgr slice_mgr_;
  std::unique_ptr<GlobalWorkQueue<std::string>> global_slice_queue_;
  DataState data_state_;
  std::unique_ptr<std::thread> init_thread_;
  Status mgr_status_;
  mutable std::mutex mu_;
};

} // End of namespace deeprecmaster
