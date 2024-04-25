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

#include <cstddef>
#include <string>
#include <vector>

namespace deeprecmaster {

class JsonReader {
 public:
  JsonReader(const std::string& json);
  ~JsonReader();

  operator bool() const { return !error_; }

  JsonReader& StartObject();
  JsonReader& Member(const std::string& name);
  bool HasMember(const std::string& name) const;
  JsonReader& EndObject();

  JsonReader& StartArray(size_t& size);
  JsonReader& EndArray();

  JsonReader& Jsonize(bool& b);
  JsonReader& Jsonize(unsigned& u);
  JsonReader& Jsonize(int& i);
  JsonReader& Jsonize(int64_t& i);
  JsonReader& Jsonize(double& d);
  JsonReader& Jsonize(std::string& s);

  template<typename T>
  bool ReadVar(const std::string &name, T &var) {
    if (HasMember(name.c_str())) {
      Member(name.c_str());
      return Jsonize(var);
    }
    return false;
  }

  template<typename T>
  bool ReadVecVar(const std::string &name, std::vector<T> &vec_var) {
    if (HasMember(name.c_str())) {
      Member(name.c_str());
      size_t count = 0;
      StartArray(count);
      bool is_ok = true;
      T value;
      for (size_t i = 0; i < count; ++i) {
        if (!Jsonize(value)) {
          is_ok = false;
          break;
        }
        vec_var.push_back(value);
      }
      EndArray();
      return is_ok;
    }
    return false;
  }

 private:
  JsonReader(const JsonReader&);
  JsonReader& operator=(const JsonReader&);

  void next();

  // PIMPL
  void* document_;              ///< DOM result of parsing.
  void* stack_;                 ///< Stack for iterating the DOM
  bool error_;                  ///< Whether an error has occurred.
};

template<typename T>
bool FromJson(T& var, JsonReader& json_reader) {
  json_reader.StartObject();
  var.fromJson(json_reader);
  json_reader.EndObject();
  return json_reader;
}


class JsonWriter {
 public:
  JsonWriter();
  ~JsonWriter();

  const char* GetString() const;

  operator bool() const { return true; }

  JsonWriter& StartObject();
  JsonWriter& Member(const std::string& name);
  JsonWriter& EndObject();

  JsonWriter& StartArray();
  JsonWriter& EndArray();

  JsonWriter& Jsonize(bool b);
  JsonWriter& Jsonize(unsigned u);
  JsonWriter& Jsonize(int i);
  JsonWriter& Jsonize(int64_t i);
  JsonWriter& Jsonize(double d);
  JsonWriter& Jsonize(const std::string& s);
  JsonWriter& SetNull();

  template<typename T>
  bool WriteVar(const std::string& name, const T& var) {
    Member(name.c_str());
    Jsonize(var);
    return true;
  }

  template<typename T>
  bool WriteVecVar(const std::string& name, const std::vector<T>& vec_var) {
    Member(name.c_str());
    size_t count = vec_var.size();
    StartArray();
    for (size_t i = 0; i < count; ++i) {
      T value = vec_var[i];
      Jsonize(value);
    }
    EndArray();
    return true;
  }

 private:
  JsonWriter(const JsonWriter&);
  JsonWriter& operator=(const JsonWriter&);

  // PIMPL idiom
  void* writer_;      ///< JSON writer.
  void* stream_;      ///< Stream buffer.
};

template<typename T>
bool ToJson(T& var, JsonWriter& json_writer, std::string& result) {
  json_writer.StartObject();
  var.toJson(json_writer);
  json_writer.EndObject();
  result = json_writer.GetString();
  return json_writer;
}

} // End of namespace deeprecmaster
