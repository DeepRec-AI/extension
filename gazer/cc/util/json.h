#ifndef AIMASTER_CORE_JSON_H_
#define AIMASTER_CORE_JSON_H_

#include <cstddef>
#include <string>
#include <vector>

namespace gazer {

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

} // namespace gazer

#endif // AIMASTER_CORE_JSON_H_