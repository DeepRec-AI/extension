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

// This function is a copy of tensorflow::DataFilename.
std::string CKPTDataFilename(StringPiece cache_path_prefix, int32 shard_id,
                             int32 num_shards);

std::string CacheCKPTDataFilename(StringPiece ckpt_path_prefix,
                                  StringPiece cache_ckpt_path, int32 shard_id,
                                  int32 num_shards);

std::string GenerateCacheCKPTKey(StringPiece ckpt_path_prefix, int32 shard_id,
                                 int32 num_shards);

std::string CacheCKPTDataFilenamePattern(StringPiece ckpt_path_prefix,
                                         StringPiece cache_ckpt_path,
                                         int32 shard_id, int32 num_shards);

std::string CacheCKPTMetaFilenamePattern(StringPiece ckpt_path_prefix,
                                         StringPiece cache_ckpt_path);

} // End of namespace tensorflow

#endif // End of TF_FAULT_TOLERANCE_CC_UTILS_CACHE_CKPT_NAMING_H_
