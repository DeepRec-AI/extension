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

#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace deeprecmaster {

class DataSliceMgr {
 public:
  DataSliceMgr();
  ~DataSliceMgr();
  void AddSlice(
      const std::string& slice_tag,
      const std::vector<std::string>& slice_data,
      int64_t slice_size);
  void AddSlice(
      const std::string& slice_tag,
      std::vector<std::string>&& slice_data,
      int64_t slice_size);
  void GetAllSliceTag(std::vector<std::string>* slice_tag);
  bool GetSliceData(const std::string& slice_tag,
                    std::vector<std::string>* slice_data);
  int64_t GetSliceCount();
  void AddSlicePrefix(const std::string& slice_tag,
                      const std::string& slice_prefix);
  bool GetSlicePrefix(const std::string& slice_tag, std::string* slice_prefix);
  int64_t GetSliceSize(const std::string& slice_tag);
  void Clear();

 private:
  std::set<std::string> slice_tag_;
  std::unordered_map<std::string, int64_t> slice_size_;
  std::unordered_map<std::string, std::vector<std::string>> slice_data_;
  std::unordered_map<std::string, int64_t> slice_tag_prefix_; // slice_tag -> slice_prefix_index
  std::vector<std::string> slice_prefix_;
  std::unordered_map<std::string, int64_t> slice_prefix_index_; // slice_prefix -> slice_prefix_index
};

} // End of namespace deeprecmaster
