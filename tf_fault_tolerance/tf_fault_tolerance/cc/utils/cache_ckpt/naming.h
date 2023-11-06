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

#ifndef TF_FAULT_TOLERANCE_CC_UTILS_CACHE_CKPT_NAMING_H_
#define TF_FAULT_TOLERANCE_CC_UTILS_CACHE_CKPT_NAMING_H_

#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {

/**
 * @brief Get CKPTFilenamePrefix without global_step.
 *
 * @param [in] ckpt_file_path_prefix: ckpt path prefix with global_step,
 *               format: '/path/to/ckpt/dir/model.ckpt-<global_step>'
 *
 * @return ckpt path prefix without global_step,
 *           format: '/path/to/ckpt/dir/model.ckpt'
 */
std::string CKPTPathPrefixWithoutGlobalStep(
              const std::string& ckpt_file_path_prefix);

/**
 * @brief Get cache ckpt file path prefix.
 *
 * @param [in] cache_path: path to save cache ckpt file.
 * @param [in] ckpt_filename_prefix: filename prefix without global_step,
 *               for example: 'model.ckpt'
 * @param [in] global_step: global_step of this ckpt.
 *
 * @return cache ckpt path prefix with global step,
 *           format: '/cache/path/model.ckpt-<global_step>'
 */
std::string CacheCKPTFilePathPrefix(StringPiece cache_path,
              StringPiece ckpt_filename_prefix, const int64 global_step);

// This function is a copy of tensorflow::MetaFilename.
std::string CKPTMetaFilename(StringPiece ckpt_path_prefix);

/**
 * @brief Get cache ckpt meta file path.
 *
 * @param [in] ckpt_path_prefix: filename prefix, for example:
                 '/cache/path/model.ckpt-<global_step>'
 * @param [in] shard: shard id of this ckpt.
 * @param [in] num_shards: num_shards of this ckpt.
 *
 * @return cache ckpt meta file
 *           format: '/cache/path/model.ckpt-<global_step>.index-<shard>-of-<num_shard>'
 */
std::string CacheCKPTMetaFilename(StringPiece ckpt_path_prefix,
                                  const int32 shard, const int32 num_shards);

// This function is a copy of tensorflow::DataFilename.
std::string CKPTDataFilename(StringPiece ckpt_path_prefix, int32 shard,
                             int32 num_shards);

std::string GenerateCKPTKey(StringPiece ckpt_path_prefix, int32 shard,
                            int32 num_shards);

void ParseCKPTKey(const std::string& ckpt_key, std::string& shard_key,
                  int64& global_step);

void ParseShardKey(const std::string& shard_key, int32& shard,
                   int32& num_shards);

} // End of namespace tensorflow

#endif // End of TF_FAULT_TOLERANCE_CC_UTILS_CACHE_CKPT_NAMING_H_
