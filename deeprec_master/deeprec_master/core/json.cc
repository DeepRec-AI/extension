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

#include "deeprec_master/include/json.h"

#include <cassert>
#include <stack>

#include "rapidjson/document.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"

namespace deeprecmaster {

using namespace rapidjson;

struct JsonReaderStackItem {
  enum State {
    BeforeStart,    //!< An object/array is in the stack but it is not yet called by StartObject()/StartArray().
    Started,        //!< An object/array is called by StartObject()/StartArray().
    Closed          //!< An array is closed after read all element, but before EndArray().
  };

  JsonReaderStackItem(const Value* value, State state) : value(value), state(state), index() {}

  const Value* value;
  State state;
  SizeType index;   // For array iteration
};

typedef std::stack<JsonReaderStackItem> JsonReaderStack;

#define DOCUMENT reinterpret_cast<Document*>(document_)
#define STACK (reinterpret_cast<JsonReaderStack*>(stack_))
#define TOP (STACK->top())
#define CURRENT (*TOP.value)

JsonReader::JsonReader(const std::string& json) : document_(), stack_(), error_(false) {
  document_ = new Document;
  DOCUMENT->Parse(json.c_str());
  if (DOCUMENT->HasParseError())
    error_ = true;
  else {
    stack_ = new JsonReaderStack;
    STACK->push(JsonReaderStackItem(DOCUMENT, JsonReaderStackItem::BeforeStart));
  }
}

JsonReader::~JsonReader() {
  delete DOCUMENT;
  delete STACK;
}

// Archive concept
JsonReader& JsonReader::StartObject() {
  if (!error_) {
    if (CURRENT.IsObject() && TOP.state == JsonReaderStackItem::BeforeStart)
      TOP.state = JsonReaderStackItem::Started;
    else
      error_ = true;
  }
  return *this;
}

JsonReader& JsonReader::EndObject() {
  if (!error_) {
    if (CURRENT.IsObject() && TOP.state == JsonReaderStackItem::Started)
      next();
    else
      error_ = true;
  }
  return *this;
}

JsonReader& JsonReader::Member(const std::string& name) {
  if (!error_) {
    if (CURRENT.IsObject() && TOP.state == JsonReaderStackItem::Started) {
      Value::ConstMemberIterator MemberItr = CURRENT.FindMember(name.c_str());
      if (MemberItr != CURRENT.MemberEnd())
        STACK->push(JsonReaderStackItem(&MemberItr->value, JsonReaderStackItem::BeforeStart));
      else
        error_ = true;
    }
    else
      error_ = true;
  }
  return *this;
}

bool JsonReader::HasMember(const std::string& name) const {
  if (!error_ && CURRENT.IsObject() && TOP.state == JsonReaderStackItem::Started)
    return CURRENT.HasMember(name.c_str());
  return false;
}

JsonReader& JsonReader::StartArray(size_t& size) {
  if (!error_) {
    if (CURRENT.IsArray() && TOP.state == JsonReaderStackItem::BeforeStart) {
      TOP.state = JsonReaderStackItem::Started;
      size = CURRENT.Size();

      if (!CURRENT.Empty()) {
        const Value* value = &CURRENT[TOP.index];
        STACK->push(JsonReaderStackItem(value, JsonReaderStackItem::BeforeStart));
      }
      else
        TOP.state = JsonReaderStackItem::Closed;
    }
    else
      error_ = true;
  }
  return *this;
}

JsonReader& JsonReader::EndArray() {
  if (!error_) {
    if (CURRENT.IsArray() && TOP.state == JsonReaderStackItem::Closed)
      next();
    else
      error_ = true;
  }
  return *this;
}

JsonReader& JsonReader::Jsonize(bool& b) {
  if (!error_) {
    if (CURRENT.IsBool()) {
      b = CURRENT.GetBool();
      next();
    }
    else
      error_ = true;
  }
  return *this;
}

JsonReader& JsonReader::Jsonize(unsigned& u) {
  if (!error_) {
    if (CURRENT.IsUint()) {
      u = CURRENT.GetUint();
      next();
    }
    else
      error_ = true;
  }
  return *this;
}

JsonReader& JsonReader::Jsonize(int& i) {
  if (!error_) {
    if (CURRENT.IsInt()) {
      i = CURRENT.GetInt();
      next();
    }
    else
      error_ = true;
  }
  return *this;
}

JsonReader& JsonReader::Jsonize(int64_t& i) {
  if (!error_) {
    if (CURRENT.IsInt64()) {
      i = CURRENT.GetInt64();
      next();
    }
    else
      error_ = true;
  }
  return *this;
}

JsonReader& JsonReader::Jsonize(double& d) {
  if (!error_) {
    if (CURRENT.IsNumber()) {
      d = CURRENT.GetDouble();
      next();
    }
    else
      error_ = true;
  }
  return *this;
}

JsonReader& JsonReader::Jsonize(std::string& s) {
  if (!error_) {
    if (CURRENT.IsString()) {
      s = CURRENT.GetString();
      next();
    }
    else
      error_ = true;
  }
  return *this;
}

void JsonReader::next() {
  if (!error_) {
    assert(!STACK->empty());
    STACK->pop();

    if (!STACK->empty() && CURRENT.IsArray()) {
      if (TOP.state == JsonReaderStackItem::Started) { // Otherwise means reading array item pass end
        if (TOP.index < CURRENT.Size() - 1) {
          const Value* value = &CURRENT[++TOP.index];
          STACK->push(JsonReaderStackItem(value, JsonReaderStackItem::BeforeStart));
        }
        else
          TOP.state = JsonReaderStackItem::Closed;
      }
      else
        error_ = true;
    }
  }
}

#undef DOCUMENT
#undef STACK
#undef TOP
#undef CURRENT


#define WRITER reinterpret_cast<PrettyWriter<StringBuffer>*>(writer_)
#define STREAM reinterpret_cast<StringBuffer*>(stream_)

JsonWriter::JsonWriter() : writer_(), stream_() {
  stream_ = new StringBuffer;
  writer_ = new PrettyWriter<StringBuffer>(*STREAM);
}

JsonWriter::~JsonWriter() {
  delete WRITER;
  delete STREAM;
}

const char* JsonWriter::GetString() const {
  return STREAM->GetString();
}

JsonWriter& JsonWriter::StartObject() {
  WRITER->StartObject();
  return *this;
}

JsonWriter& JsonWriter::EndObject() {
  WRITER->EndObject();
  return *this;
}

JsonWriter& JsonWriter::Member(const std::string& name) {
  WRITER->String(name.c_str(), static_cast<SizeType>(name.size()));
  return *this;
}

JsonWriter& JsonWriter::StartArray() {
  WRITER->StartArray();
  return *this;
}

JsonWriter& JsonWriter::EndArray() {
  WRITER->EndArray();
  return *this;
}

JsonWriter& JsonWriter::Jsonize(bool b) {
  WRITER->Bool(b);
  return *this;
}

JsonWriter& JsonWriter::Jsonize(unsigned u) {
  WRITER->Uint(u);
  return *this;
}

JsonWriter& JsonWriter::Jsonize(int i) {
  WRITER->Int(i);
  return *this;
}

JsonWriter& JsonWriter::Jsonize(int64_t i) {
  WRITER->Int64(i);
  return *this;
}

JsonWriter& JsonWriter::Jsonize(double d) {
  WRITER->Double(d);
  return *this;
}

JsonWriter& JsonWriter::Jsonize(const std::string& s) {
  WRITER->String(s.c_str(), static_cast<SizeType>(s.size()));
  return *this;
}

JsonWriter& JsonWriter::SetNull() {
  WRITER->Null();
  return *this;
}

#undef STREAM
#undef WRITER

} // End of namespace deeprecmaster
