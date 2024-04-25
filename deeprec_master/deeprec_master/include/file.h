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

#include <string>
#include <unistd.h>
#include <vector>

#include "deeprec_master/include/status.h"

namespace deeprecmaster {

struct FileStatistics {
  // The length of the file or -1 if finding file length is not supported.
  int64_t length = -1;
  // The last modified time in nanoseconds.
  int64_t mtime_nsec = 0;
  // True if the file is a directory, otherwise false.
  bool is_directory = false;

  FileStatistics() {}
  FileStatistics(int64_t length, int64_t mtime_nsec, bool is_directory)
      : length(length), mtime_nsec(mtime_nsec), is_directory(is_directory) {}
  ~FileStatistics() {}
};

// Collapse duplicate "/"s, resolve ".." and "." path elements, remove
// trailing "/".
//
// NOTE: This respects relative vs. absolute paths, but does not
// invoke any system calls (getcwd(2)) in order to resolve relative
// paths with respect to the actual working directory.  That is, this is purely
// string manipulation, completely independent of process state.
std::string FormatPath(const std::string& path);

// Returns OK if the named path exists and NOT_FOUND otherwise.
Status FileExists(const std::string& fname);

// Obtains statistics for the given path
Status Stat(const std::string& fname, FileStatistics& stats);

// Returns whether the given path is a directory or not.
//
// Typical return codes (not guaranteed exhaustive):
//  * OK - The path exists and is a directory.
//  * FAILED_PRECONDITION - The path exists and is not a directory.
//  * NOT_FOUND - The path entry does not exist.
//  * PERMISSION_DENIED - Insufficient permissions.
//  * UNIMPLEMENTED - The file factory doesn't support directories.
Status IsDirectory(const std::string& fname);

// Returns the immediate children in the given directory.
// The returned paths are relative to 'dir'.
Status GetChildren(const std::string& dir, std::vector<std::string>& result);

} // End of namespace deeprecmaster
