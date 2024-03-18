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
#include <sys/time.h>
#include <unistd.h>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>

#include "deeprec_master/include/logging.h"

namespace deeprecmaster {

Logger& Logger::Instance() {
  static Logger g_log;
  return g_log;
}

Logger::Logger() : output(stderr) {
  const char* env_value = std::getenv("AIMASTER_LOG_LEVEL");
  log_level = env_value ? Logger::ParseLevel(env_value) : Logger::INFO;
  // gethostname(hostname, sizeof(hostname));
}

Logger::~Logger() {
  if (output != stderr) fclose(output);
}

void Logger::SetLevel(LoggingLevel level) { log_level = level; }

bool Logger::SetOutput(const std::string& filename) {
  if (filename.empty()) return true;

  FILE* file = fopen(filename.c_str(), "w");
  if (file == NULL) return false;

  output = file;
  return true;
}

Logger::LoggingLevel Logger::ParseLevel(const std::string& str) {
  size_t count = sizeof(loggingLevelNames) / sizeof(loggingLevelNames[0]);
  for (size_t i = 0; i < count; i++) {
    if (str.compare(loggingLevelNames[i]) == 0)
      return static_cast<Logger::LoggingLevel>(i);
  }
  return Logger::NONE;
}

}  // namespace deeprecmaster