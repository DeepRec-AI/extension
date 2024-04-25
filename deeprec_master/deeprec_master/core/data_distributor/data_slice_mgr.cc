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

#include <algorithm>

#include "deeprec_master/include/data_distributor/data_slice_mgr.h"
#include "deeprec_master/include/logging.h"

namespace deeprecmaster {

DataSliceMgr::DataSliceMgr() {}

DataSliceMgr::~DataSliceMgr() {}

void DataSliceMgr::AddSlice(const std::string& slice_tag,
                            const std::vector<std::string>& slice_data,
                            int64_t slice_size) {
  slice_tag_.insert(slice_tag);
  slice_data_[slice_tag] = slice_data;
  slice_size_[slice_tag] = slice_size;
}

void DataSliceMgr::AddSlice(const std::string& slice_tag,
                            std::vector<std::string>&& slice_data,
                            int64_t slice_size) {
  slice_tag_.insert(slice_tag);
  slice_data_[slice_tag].swap(slice_data);
  slice_size_[slice_tag] = slice_size;
}

void DataSliceMgr::GetAllSliceTag(std::vector<std::string>* slice_tag) {
  std::vector<std::string> tag_vec(slice_tag_.begin(), slice_tag_.end());
  std::sort(tag_vec.begin(), tag_vec.end(),
      [](const std::string& tag1, const std::string& tag2) {
        if (tag1.size() != tag2.size()) {
          return tag1.size() < tag2.size();
        }
        return tag1 < tag2;
      });
  slice_tag->swap(tag_vec);
}

bool DataSliceMgr::GetSliceData(const std::string& slice_tag,
                                std::vector<std::string>* slice_data) {
  if (slice_tag_.find(slice_tag) == slice_tag_.end()) {
    return false;
  }
  *slice_data = slice_data_[slice_tag];
  return true;
}

int64_t DataSliceMgr::GetSliceCount() {
  return slice_tag_.size();
}

void DataSliceMgr::AddSlicePrefix(const std::string& slice_tag,
                                  const std::string& slice_prefix) {
  int64_t prefix_index = 0;
  auto prefix_index_iter = slice_prefix_index_.find(slice_prefix);
  if (prefix_index_iter == slice_prefix_index_.end()) {
    slice_prefix_.push_back(slice_prefix);
    prefix_index = slice_prefix_.size() - 1;
    slice_prefix_index_[slice_prefix] = prefix_index;
  } else {
    prefix_index = prefix_index_iter->second;
  }

  slice_tag_prefix_[slice_tag] = prefix_index;
}

bool DataSliceMgr::GetSlicePrefix(const std::string& slice_tag,
                                  std::string* slice_prefix) {
  slice_prefix->clear();
  auto iter = slice_tag_prefix_.find(slice_tag);
  if (iter == slice_tag_prefix_.end()) {
    return false;
  }

  int64_t prefix_index = iter->second;
  if (prefix_index >= slice_prefix_.size()) {
    LogError("Invalid slice prefix index %d >= %d", prefix_index, slice_prefix_.size());
    return false;
  }

  *slice_prefix = slice_prefix_[prefix_index];
  return true;
}

int64_t DataSliceMgr::GetSliceSize(const std::string& slice_tag) {
  const auto& iter = slice_size_.find(slice_tag);
  if (iter == slice_size_.end()) {
    return -1;
  }

  return iter->second;
}

void DataSliceMgr::Clear() {
  slice_tag_.clear();
  slice_size_.clear();
  slice_data_.clear();
  slice_tag_prefix_.clear();
  slice_prefix_.clear();
  slice_prefix_index_.clear();
}

} // End of namespace deeprecmaster
