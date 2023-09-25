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

#include <vector>
#include <string>

#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/default/logging.h"

#include "tf_fault_tolerance/cc/utils/cache_ckpt/naming.h"

namespace tensorflow {

std::string CKPTDataFilename(StringPiece cache_path_prefix, int32 shard_id,
                             int32 num_shards) {
  CHECK_GT(num_shards, 0);
  CHECK_LT(shard_id, num_shards);
  return strings::Printf("%.*s.data-%05d-of-%05d",
                         static_cast<int>(cache_path_prefix.size()),
                         cache_path_prefix.data(),
                         shard_id, num_shards);
}

std::string CacheCKPTDataFilename(StringPiece ckpt_path_prefix,
                                  StringPiece cache_ckpt_path, int32 shard_id,
                                  int32 num_shards) {
  CHECK_GT(num_shards, 0);
  CHECK_LT(shard_id, num_shards);

  std::vector<std::string> ckpt_path_prefix_vec = \
    absl::StrSplit(ckpt_path_prefix, "/");

  std::string ckpt_data_file_prefix = ckpt_path_prefix_vec.back();
  std::string cache_ckpt_data_file_prefix = \
    absl::StrCat(cache_ckpt_path, "/", ckpt_data_file_prefix);

  return strings::Printf("%.*s.data-%05d-of-%05d",
                         static_cast<int>(cache_ckpt_data_file_prefix.size()),
                         cache_ckpt_data_file_prefix.data(),
                         shard_id, num_shards);
}

std::string GenerateCacheCKPTKey(StringPiece ckpt_path_prefix, int32 shard_id,
                                 int32 num_shards) {
  std::vector<std::string> ckpt_path_prefix_vec = \
    absl::StrSplit(ckpt_path_prefix, "-");
  std::string global_step = ckpt_path_prefix_vec.back();

  return strings::Printf("%.*s:%05d-%05d", static_cast<int>(global_step.size()),
                         global_step.data(), shard_id, num_shards);
}

std::string CacheCKPTDataFilenamePattern(StringPiece ckpt_path_prefix,
                                         StringPiece cache_ckpt_path,
                                         int32 shard_id, int32 num_shards) {
  CHECK_GT(num_shards, 0);
  CHECK_LT(shard_id, num_shards);

  std::vector<std::string> ckpt_path_prefix_vec = \
    absl::StrSplit(ckpt_path_prefix, "/");

  std::string ckpt_data_file_prefix = ckpt_path_prefix_vec.back();
  // rm global step.
  std::vector<std::string> ckpt_data_file_prefix_vec = \
    absl::StrSplit(ckpt_data_file_prefix, "-");
  ckpt_data_file_prefix = ckpt_data_file_prefix_vec.front();

  std::string cache_ckpt_data_file_prefix = \
    absl::StrCat(cache_ckpt_path, "/", ckpt_data_file_prefix);

  return strings::Printf("%.*s-[0-9]*.data-%05d-of-%05d",
                         static_cast<int>(cache_ckpt_data_file_prefix.size()),
                         cache_ckpt_data_file_prefix.data(),
                         shard_id, num_shards);
}

std::string CacheCKPTMetaFilenamePattern(StringPiece ckpt_path_prefix,
                                         StringPiece cache_ckpt_path) {
  std::vector<std::string> ckpt_path_prefix_vec = \
    absl::StrSplit(ckpt_path_prefix, "/");

  std::string ckpt_meta_file_prefix = ckpt_path_prefix_vec.back();
  // rm global step.
  std::vector<std::string> ckpt_meta_file_prefix_vec = \
    absl::StrSplit(ckpt_meta_file_prefix, "-");
  ckpt_meta_file_prefix = ckpt_meta_file_prefix_vec.front();

  std::string cache_ckpt_meta_file_prefix = \
    absl::StrCat(cache_ckpt_path, "/", ckpt_meta_file_prefix);

  return strings::Printf("%.*s-[0-9]*.index",
                         static_cast<int>(cache_ckpt_meta_file_prefix.size()),
                         cache_ckpt_meta_file_prefix.data());
}

} // End of namespace tensorflow
