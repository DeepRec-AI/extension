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

#include <fstream>
#include <string>
#include <vector>

#include "absl/strings/escaping.h"
#include "absl/strings/str_format.h"

#include "deeprec_master/include/data_distributor/data_manager.h"
#include "deeprec_master/include/errors.h"
#include "deeprec_master/include/file.h"
#include "deeprec_master/include/json.h"
#include "deeprec_master/include/logging.h"
#include "deeprec_master/include/utils.h"

namespace deeprecmaster {

namespace {
std::string GenTaskId(const std::string& task_name, int32_t task_index) {
  return task_name + "#" + std::to_string(task_index);
}
} // End of anonymous namespace

//------------------------------------------------------------------------------
// Functions of class DataManagerConfig.
DataManagerConfig::DataManagerConfig() {}

Status DataManagerConfig::ParseDataConfigFromString(
         const std::string& config_pb) {
  std::string config_pb_str;
  absl::Base64Unescape(config_pb, &config_pb_str);
  bool is_ok = data_config_.ParseFromString(config_pb_str);
  if (!is_ok) {
    std::string emsg = "Failed to parse data config from pb string.";
    LogError(emsg.c_str());
    return error::InvalidArgument(emsg);
  }
  LogInfo("DataConfig: %s", data_config_.DebugString());
  return Status::OK();
}

Status DataManagerConfig::ValidatePosixFileDataConfig() {
  const auto& file_config = data_config_.file_config();
  for (int idx = 0; idx < file_config.input_dirs_size(); ++idx) {
    const std::string& input_dir = file_config.input_dirs(idx);
    Status st = IsDirectory(input_dir);
    if (!st.ok()) {
      std::string emsg = \
        absl::StrFormat("Invalid input_dirs, %s not a directory, %s",
                        input_dir, st.ToString());
      return error::InvalidArgument(emsg);
    }
  }

  const std::string& file_meta_path = file_config.input_file_meta_path();
  if (!file_meta_path.empty()) {
    if (!FileExists(file_meta_path).ok()) {
      return error::InvalidArgument(
          "Invalid file_meta_path, " + file_meta_path + " not exist.");
    }
    if (IsDirectory(file_meta_path).ok()) {
      return error::InvalidArgument(
          "Invalid file_meta_path, " + file_meta_path + " is a directory.");
    }
  }

  return Status::OK();
}

Status DataManagerConfig::ValidateDataConfig() {
  if (data_config_.dstype() == DataConfig::POSIX_FILE) {
    return ValidatePosixFileDataConfig();
  }

  std::string emsg = absl::StrFormat(
      "Not support %d datasource.", data_config_.dstype());
  LogError(emsg.c_str());
  return error::Unimplemented(emsg);
}

const DataConfig& DataManagerConfig::GetDataConfig() const {
  return data_config_;
}

//------------------------------------------------------------------------------
// Functions of class DataManager.
DataManager::DataManager()
  : is_ready_(false), continue_dispatch_data_(false), total_slice_count_(0),
    processed_slice_count_(0), mgr_status_(Status::OK()) {}

DataManager::~DataManager() {
  if(init_thread_ && init_thread_->joinable()) {
    init_thread_->join();
  }
}

Status DataManager::Init(const std::string& task_name, const int32_t task_index,
                         const std::string& config_pb) {
  std::lock_guard<std::mutex> lock(mu_);
  RETURN_IF_ERROR(mgr_status_);
  Clear();
  mgr_status_ = mgr_config_.ParseDataConfigFromString(config_pb);
  RETURN_IF_ERROR(mgr_status_);
  mgr_status_ = mgr_config_.ValidateDataConfig();
  RETURN_IF_ERROR(mgr_status_);
  data_state_.Init(mgr_config_.GetDataConfig());
  init_thread_.reset(new std::thread(&DataManager::DoInit, this));

  return Status::OK();
}

Status DataManager::IsReady(const std::string& task_name,
                            const int32_t task_index, bool* is_ready) {
  std::lock_guard<std::mutex> lock(mu_);
  *is_ready = is_ready_;
  RETURN_IF_ERROR(mgr_status_);
  return Status::OK();
}

bool DataManager::IsReady() {
  std::lock_guard<std::mutex> lock(mu_);
  return is_ready_;
}

Status DataManager::GetSlice(const std::string& task_name,
                             const int32_t task_index, std::string* slice_tag,
                             std::string* slice_prefix, int64_t* slice_size,
                             std::vector<std::string>* slice_data) {
  slice_tag->clear();
  slice_prefix->clear();
  slice_data->clear();
  const std::string& task_id = GenTaskId(task_name, task_index);
  int64_t now_timestamp = GetTimeStamp(1);

  std::lock_guard<std::mutex> lock(mu_);
  if (!is_ready_) {
    return error::Unavailable("DataManager not ready.");
  }
  if (!continue_dispatch_data_) {
    return error::RetryLater(
             "DataManager stop dispatching data, please retry later.");
  }

  if (!global_slice_queue_->IsEmpty()) {
    global_slice_queue_->Front(*slice_tag);
    global_slice_queue_->Pop();
    UpdateDataState();
  }

  if (slice_tag->empty()) {
    LogError("No more slice to get, %s", task_id);
    return error::OutOfRange("Out of range.");
  }

  bool is_ok = slice_mgr_.GetSliceData(*slice_tag, slice_data);
  if (!is_ok) {
    throw std::runtime_error("Not found slice data for slice_tag " + *slice_tag);
  }
  slice_mgr_.GetSlicePrefix(*slice_tag, slice_prefix);
  *slice_size = slice_mgr_.GetSliceSize(*slice_tag);
  processed_slice_count_ += 1;
  int64_t take_cost = GetTimeStamp(0) - now_timestamp;
  LogInfo("Get slice %s, %s, cost %ldµs", *slice_tag, task_id, take_cost);

  return Status::OK();
}

Status DataManager::GetDataState(const std::string& task_name,
                                 const int32_t task_index,
                                 std::string* data_state) {
  std::lock_guard<std::mutex> lock(mu_);
  if (!data_state_.SerializeToBase64String(data_state)) {
    return error::Internal("Failed to get data state.");
  }
  return Status::OK();
}

Status DataManager::StartDataDispatch(const std::string& task_name,
                                      const int32_t task_index) {
  std::lock_guard<std::mutex> lock(mu_);
  continue_dispatch_data_ = true;

  return Status::OK();
}

Status DataManager::StopDataDispatch(const std::string& task_name,
                                     const int32_t task_index) {
  std::lock_guard<std::mutex> lock(mu_);
  continue_dispatch_data_ = false;

  return Status::OK();
}

void DataManager::Clear() {
  is_ready_ = false;
  continue_dispatch_data_ = false;
  total_slice_count_ = 0;
  processed_slice_count_ = 0;
  slice_mgr_.Clear();
  if (init_thread_ && init_thread_->joinable()) {
    init_thread_->join();
  }
  mgr_status_ = Status::OK();
  // INFO: mgr_config_, global_slice_queue_ and data_state will be reinitialized
  // later.
}

void DataManager::UpdateDataState() {
  int64_t cur_epoch = global_slice_queue_->GetCurrentEpoch();
  int64_t shuffle_seed = global_slice_queue_->GetCurrentShuffleSeed();
  int64_t slice_index = global_slice_queue_->GetCurrentSliceIndex();
  data_state_.UpdateEpochIndex(cur_epoch);
  data_state_.UpdateShuffleSeed(shuffle_seed);
  data_state_.UpdateSliceIndex(slice_index);
}

Status DataManager::DoInit() {
  LogInfo("Initializing data manager ...");
  int64_t start_time = GetTimeStamp();
  const auto& data_config = mgr_config_.GetDataConfig();
  Status init_status;

  if (data_config.dstype() == DataConfig::POSIX_FILE) {
    if (!data_config.file_config().input_file_meta_path().empty()) {
      init_status = DoInitFromFileMeta();
    } else {
      init_status = DoInitFromFileDir();
    }
  } else {
    init_status = error::Unimplemented("Unsupported datasource.");
  }

  if (init_status.ok()) {
    init_status = DoInitGlabalSliceQueue();
  }

  {
    std::lock_guard<std::mutex> lock(mu_);
    mgr_status_ = init_status;
    RETURN_IF_ERROR(init_status);
    total_slice_count_ = global_slice_queue_->Size();
    is_ready_ = true;
    continue_dispatch_data_ = true;
  }
  LogInfo("Total slice count %lld", total_slice_count_);
  int64_t finish_time = GetTimeStamp();
  float cost_time = (finish_time - start_time) / 1000000.0;
  LogInfo("Initialized data manager, cost %.2fs", cost_time);
  return Status::OK();
}

Status DataManager::DoInitFromFileMeta() {
  const auto& file_config = mgr_config_.GetDataConfig().file_config();
  std::string file_meta_path = file_config.input_file_meta_path();
  int64_t file_slice_size = file_config.slice_size();

  std::ifstream file_in(file_meta_path);
  if (!file_in.good()) {
    throw
      std::runtime_error("Failed to read data config from " + file_meta_path);
  }
  std::string meta_info((std::istreambuf_iterator<char>(file_in)),
                        std::istreambuf_iterator<char>());

  int64_t file_count = 0;
  std::string file_dir;
  std::vector<std::string> input_files;
  JsonReader reader(meta_info);
  reader.StartObject();
  if (!reader) {
    return error::InvalidArgument("Invalid json file " + file_meta_path);
  }
  if (!reader.ReadVar("FileCount", file_count)) {
    return error::InvalidArgument("Not found FileCount in " + file_meta_path);
  }
  reader.ReadVar("FileDir", file_dir);
  if (!reader.ReadVecVar("FileList", input_files)) {
    return error::InvalidArgument("Not found FileList in " + file_meta_path);
  }
  reader.EndObject();

  if (file_count != input_files.size()) {
    std::string emsg = absl::StrFormat(
          "FileList size should be equal FileCount in %s, not %lld vs %lld",
          file_meta_path, input_files.size(), file_count);
    LogError(emsg.c_str());
    return error::InvalidArgument(emsg);
  }

  LogInfo("file_meta_path: %s, file count: %lld", file_meta_path, file_count);

  int64_t slice_id = 0;
  for (auto slice_begin = input_files.begin(); slice_begin != input_files.end(); ) {
    auto slice_end = (slice_begin + file_slice_size < input_files.end())
                    ? (slice_begin + file_slice_size) : input_files.end();
    std::vector<std::string> slice_data(slice_begin, slice_end);
    std::string slice_tag = std::to_string(slice_id);
    int64_t slice_size = slice_end - slice_begin;
    slice_mgr_.AddSlice(slice_tag, std::move(slice_data), slice_size);
    slice_mgr_.AddSlicePrefix(slice_tag, file_dir);
    slice_begin = slice_end;
    ++slice_id;
  }

  return Status::OK();
}

Status DataManager::DoInitFromFileDir() {
  const auto& file_config = mgr_config_.GetDataConfig().file_config();
  int64_t file_slice_size = file_config.slice_size();
  int64_t slice_id = 0;
  for (int idx = 0; idx < file_config.input_dirs_size(); ++idx) {
    const std::string& input_dir = file_config.input_dirs(idx);
    std::vector<std::string> files;
    auto st = GetChildren(input_dir, files);
    if (!st.ok()) {
      std::string emsg = absl::StrFormat(
          "Failed to list files from dir %s, %s", input_dir, st.ToString());
      LogError(emsg.c_str());
      return error::InvalidArgument(emsg);
    }

    LogInfo("Input_dir: %s, file_count: %lld", input_dir, files.size());

    for (auto slice_begin = files.begin(); slice_begin != files.end(); ) {
      auto slice_end = (slice_begin + file_slice_size < files.end())
                      ? (slice_begin + file_slice_size) : files.end();
      std::vector<std::string> slice_data(slice_begin, slice_end);
      std::string slice_tag = std::to_string(slice_id);
      int64_t slice_size = slice_end - slice_begin;
      slice_mgr_.AddSlice(slice_tag, std::move(slice_data), slice_size);
      slice_mgr_.AddSlicePrefix(slice_tag, input_dir);
      slice_begin = slice_end;
      ++slice_id;
    }
  }

  return Status::OK();
}

Status DataManager::DoInitGlabalSliceQueue() {
  std::vector<std::string> slice_tags;
  slice_mgr_.GetAllSliceTag(&slice_tags);
  const auto& options = mgr_config_.GetDataConfig().options();

  global_slice_queue_.reset(
    new GlobalWorkQueue<std::string>(slice_tags, options.num_epochs(),
          options.shuffle(), options.shuffle_seed(), options.epoch_index(),
          options.slice_index()));

  return Status::OK();
}

DataManager* DataManager::GetInstance() {
  static DataManager* sg_mamager_ptr = new DataManager();
  return sg_mamager_ptr;
}

} // End of namespace deeprecmaster
