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

#ifndef TF_FAULT_TOLERANCE_CC_UTILS_CACHE_CKPT_STORAGE_TYPE_H_
#define TF_FAULT_TOLERANCE_CC_UTILS_CACHE_CKPT_STORAGE_TYPE_H_

#include <string>

namespace tensorflow {

// StorageType define.
namespace StorageType {
  const std::string kMemoryType = "memory";
  const std::string kPosixFileType = "filesystem";
} // End of namespace StorageType

std::string GetLocalCacheCKPTStorageTypeFromEnvVar();

std::string GetRemoteCacheCKPTStorageTypeFromEnvVar();

} // End of namespace tensorflow

#endif // End of TF_FAULT_TOLERANCE_CC_UTILS_CACHE_CKPT_STORAGE_TYPE_H_
