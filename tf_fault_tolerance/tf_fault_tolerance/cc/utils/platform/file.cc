/* Copyright 2023 The DeepRec Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/lib/strings/str_util.h"

#include "tf_fault_tolerance/cc/utils/platform/file.h"

namespace tensorflow {

std::string GeneratePOSIXFilePath(const std::string& dir,
                                  const std::string& filename) {
  return absl::StrCat(dir, "/", filename);
}

void ParsePOSIXFilePath(const std::string& file_path, std::string& dir,
                        std::string& filename) {
  auto split_index = file_path.rfind("/");
  dir = file_path.substr(0, split_index);
  filename = file_path.substr(split_index+1, file_path.size());
}

} // End of namespace tensorflow
