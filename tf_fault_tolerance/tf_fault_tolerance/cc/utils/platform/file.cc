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
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/util/env_var.h"

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

namespace io_utils {

constexpr size_t kCopyFileBufferSize = 4 * 1024 * 1024;

static bool UseTFZeroFileCopyFunc() {
  bool enable = false;
  TF_CHECK_OK(
    ReadBoolFromEnvVar("USE_TF_ZERO_FILE_ZERO_FUNC", false, &enable));

  return enable;
}

static Status FileSystemCopyFile(FileSystem* src_fs, const string& src,
                          FileSystem* target_fs, const string& target) {
  std::unique_ptr<RandomAccessFile> src_file;
  TF_RETURN_IF_ERROR(src_fs->NewRandomAccessFile(src, &src_file));

  std::unique_ptr<WritableFile> target_file;
  TF_RETURN_IF_ERROR(target_fs->NewWritableFile(target, &target_file));

  uint64 offset = 0;
  std::unique_ptr<char[]> scratch(new char[kCopyFileBufferSize]);
  Status s = Status::OK();
  while (s.ok()) {
    StringPiece result;
    s = src_file->Read(offset, kCopyFileBufferSize, &result, scratch.get());
    if (!(s.ok() || s.code() == error::OUT_OF_RANGE)) {
      return s;
    }
    TF_RETURN_IF_ERROR(target_file->Append(result));
    offset += result.size();
  }
  return target_file->Close();
}


Status CopyFile(const string& src, const string& target) {
  static bool use_tf_copy =  UseTFZeroFileCopyFunc();
  Env* env = Env::Default();
  if (use_tf_copy) {
    return env->CopyFile(src, target);
  }

  FileSystem* src_fs;
  FileSystem* target_fs;
  TF_RETURN_IF_ERROR(env->GetFileSystemForFile(src, &src_fs));
  TF_RETURN_IF_ERROR(env->GetFileSystemForFile(target, &target_fs));

  return io_utils::FileSystemCopyFile(src_fs, src, target_fs, target);
}

} // End of namespace io_utils

} // End of namespace tensorflow
