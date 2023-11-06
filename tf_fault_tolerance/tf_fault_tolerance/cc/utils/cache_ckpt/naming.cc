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

#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/default/logging.h"

#include "tf_fault_tolerance/cc/utils/cache_ckpt/naming.h"

namespace tensorflow {

std::string CKPTPathPrefixWithoutGlobalStep(
              const std::string& ckpt_path_prefix) {
  // rm global_step
  auto split_index = ckpt_path_prefix.rfind("-");
  std::string filename_prefix = ckpt_path_prefix.substr(0, split_index);
  return filename_prefix;
}

std::string CacheCKPTFilePathPrefix(StringPiece cache_path,
            StringPiece ckpt_filename_prefix, const int64 global_step) {
  std::string cache_path_prefix = \
    absl::StrCat(cache_path, "/", ckpt_filename_prefix); // without global step.

  return strings::Printf("%.*s-%lld",
                         static_cast<int>(cache_path_prefix.size()),
                         cache_path_prefix.data(), global_step);
}

std::string CKPTMetaFilename(StringPiece ckpt_path_prefix) {
  return strings::Printf("%.*s.index",
                         static_cast<int>(ckpt_path_prefix.size()),
                         ckpt_path_prefix.data());
}

std::string CacheCKPTMetaFilename(StringPiece ckpt_path_prefix,
                                  const int32 shard, const int32 num_shards) {
  return strings::Printf("%.*s.index-%05d-of-%05d",
                         static_cast<int>(ckpt_path_prefix.size()),
                         ckpt_path_prefix.data(), shard, num_shards);
}

std::string CKPTDataFilename(StringPiece ckpt_path_prefix, int32 shard,
                             int32 num_shards) {
  return strings::Printf("%.*s.data-%05d-of-%05d",
                         static_cast<int>(ckpt_path_prefix.size()),
                         ckpt_path_prefix.data(), shard, num_shards);
}

std::string GenerateCKPTKey(StringPiece ckpt_path_prefix, int32 shard,
                            int32 num_shards) {
  std::vector<std::string> ckpt_path_prefix_vec = \
    absl::StrSplit(ckpt_path_prefix, "-");
  std::string global_step = ckpt_path_prefix_vec.back();

  return strings::Printf("%.*s:%05d-%05d", static_cast<int>(global_step.size()),
                         global_step.data(), shard, num_shards);
}

void ParseCKPTKey(const std::string& ckpt_key, std::string& shard_key,
                  int64& global_step) {
  // ckpt_key format: global_step:shard-num_shards
  std::vector<std::string> ckpt_key_vec = absl::StrSplit(ckpt_key, ":");
  CHECK_EQ(ckpt_key_vec.size(), 2);

  shard_key = ckpt_key_vec.back();
  CHECK(strings::safe_strto64(ckpt_key_vec.front(), &global_step));
}

void ParseShardKey(const std::string& shard_key, int32& shard,
                   int32& num_shards) {
  // shard_key format: shard-num_shards
  std::vector<std::string> shard_key_vec = absl::StrSplit(shard_key, "-");
  CHECK_EQ(shard_key_vec.size(), 2);

  CHECK(strings::safe_strto32(shard_key_vec.front(), &shard));
  CHECK(strings::safe_strto32(shard_key_vec.back(), &num_shards));
}

} // End of namespace tensorflow
