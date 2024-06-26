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

#ifndef DYNMAMIC_EMBEDDING_SERVER_INCLUDE_UTILS_NAMING_H_
#define DYNMAMIC_EMBEDDING_SERVER_INCLUDE_UTILS_NAMING_H_

#include "dynamic_embedding_server/include/utils/tensorflow_include.h"

namespace des {

constexpr char kEvImportOp[] = "KvResourceMulImport";
constexpr char kEvExportOp[] = "KvResourceFilter";
constexpr char kReAssign[] = "ReAssign";
constexpr char kReAssignRes[] = "ReAssignResource";

}  // namespace des
#endif  // DYNMAMIC_EMBEDDING_SERVER_INCLUDE_UTILS_NAMING_H_