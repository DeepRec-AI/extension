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

#include <sys/time.h>
#include <sys/types.h>

#include <time.h>
#include <unistd.h>
#include <mutex>
#include <string>

namespace deeprecmaster {

static const char* loggingLevelNames[] = {"NONE", "ERR", "WARN", "INFO",
                                          "TRACE"};

#define Log(level, ...)                                                    \
  do {                                                                     \
    deeprecmaster::Logger::Instance().LogCommon(level, __FILE__, __LINE__, \
                                                __VA_ARGS__);              \
  } while (0)

#define LogError(...) Log(deeprecmaster::Logger::ERROR, __VA_ARGS__)
#define LogWarn(...) Log(deeprecmaster::Logger::WARN, __VA_ARGS__)
#define LogInfo(...) Log(deeprecmaster::Logger::INFO, __VA_ARGS__)

#ifdef NDEBUG
// disable TRACE log in release version
#define LogTrace(...)
#define FUNC_TRACE()
#define FUNC_TRACE_PARAMS(format, ...)
#else
#define LogTrace(...) Log(deeprecmaster::Logger::TRACE, __VA_ARGS__)
#define FUNC_TRACE()                         \
  std::string func = __PRETTY_FUNCTION__;    \
  deeprecmaster::FuncScope func_scope(func); \
  LogTrace("IN: %s", func.c_str())
#define FUNC_TRACE_PARAMS(format, ...)              \
  std::string func = __PRETTY_FUNCTION__;           \
  deeprecmaster::FuncScope func_scope_params(func); \
  LogTrace("IN: %s, " format, func.c_str(), __VA_ARGS__)
#endif

class Logger {
 public:
  enum LoggingLevel { NONE = 0, ERROR, WARN, INFO, TRACE };

  static Logger& Instance();

  Logger();
  ~Logger();

  void SetLevel(LoggingLevel level);
  bool SetOutput(const std::string& filename);

  std::string GetCurrTimeStrWithMs() {
    struct timeval now;
    gettimeofday(&now, NULL);

    struct tm ltm;
    localtime_r(&now.tv_sec, &ltm);

    char buffer[80] = {0};
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &ltm);

    char timestr[84] = {0};
    snprintf(timestr, sizeof(timestr), "%s.%03d", buffer,
             static_cast<int>(now.tv_usec / 1000));

    return timestr;
  }

  template <class T>
  T const& convert_for_log_note(T const& x) {
    return x;
  }

  const char* convert_for_log_note(std::string const& x) { return x.c_str(); }

  void get_filename(const char* filepath, std::string* filename) {
    *filename = filepath;
    auto last_slash_pos = filename->find_last_of("/\\");
    if (last_slash_pos != std::string::npos) {
      *filename = filename->substr(last_slash_pos + 1);
    }
  }

  template <class... Args>
  void LogCommon(LoggingLevel level, const char* filepath, int fileline,
                 const char* fmt, Args&&... args) {
    auto timestr = GetCurrTimeStrWithMs();
    std::string filename;
    get_filename(filepath, &filename);
    std::lock_guard<std::mutex> lock(mutex);
    fprintf(output, "[%s AIMaster] (%s %d) %s: ", timestr.c_str(),
            filename.c_str(), fileline, loggingLevelNames[level]);
    fprintf(output, fmt, convert_for_log_note(args)...);
    fprintf(output, "\n");
    fflush(output);
  }

  static LoggingLevel ParseLevel(const std::string& str);

 private:
  LoggingLevel log_level;
  FILE* output;
  std::mutex mutex;
  char hostname[256];
};

#ifndef NDEBUG
class FuncScope {
 public:
  explicit FuncScope(const std::string& func) : fname(func) {}

  ~FuncScope() { LogTrace("OUT: %s", fname.c_str()); }

 private:
  std::string fname;
};
#endif

}  // namespace deeprecmaster
