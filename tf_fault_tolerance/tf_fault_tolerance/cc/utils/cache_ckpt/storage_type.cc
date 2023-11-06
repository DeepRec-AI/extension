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

#include "tensorflow/core/util/env_var.h"

#include "tf_fault_tolerance/cc/utils/cache_ckpt/storage_type.h"

namespace tensorflow {

std::string GetLocalCacheCKPTStorageTypeFromEnvVar() {
  std::string storage_type;
  TF_CHECK_OK(ReadStringFromEnvVar("LOCAL_CACHE_CKPT_STORAGE_TYPE",
                                   StorageType::kPosixFileType, &storage_type));
  return storage_type;
}

std::string GetRemoteCacheCKPTStorageTypeFromEnvVar() {
  std::string storage_type;
  TF_CHECK_OK(ReadStringFromEnvVar("REMOTE_CACHE_CKPT_STORAGE_TYPE",
                                   StorageType::kMemoryType, &storage_type));
  return storage_type;
}

} // End of namespace tensorflow.
