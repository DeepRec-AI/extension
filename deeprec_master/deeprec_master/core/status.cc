
#include "deeprec_master/include/status.h"

#include <cstdio>
#include <cstring>

namespace deeprecmaster {

Status::Status(error::Code code, const char* msg) : msg_(nullptr) {
  Assign(code, msg);
}

Status::Status(error::Code code, const std::string& msg) : msg_(nullptr) {
  Assign(code, msg.c_str());
}

Status::Status(const Status& other) {
  code_ = other.code_;
  msg_ = CopyMessage(other.msg_);
}

Status::~Status() {
  delete[] msg_;
  msg_ = nullptr;
}

Status& Status::operator=(error::Code code) {
  code_ = code;
  delete[] msg_;
  msg_ = nullptr;
  return *this;
}

Status& Status::operator=(const Status& other) {
  if (this != &other) {
    code_ = other.code_;
    delete[] msg_;
    msg_ = CopyMessage(other.msg_);
  }
  return *this;
}

Status& Status::Assign(error::Code code, const char* msg) {
  code_ = code;
  delete[] msg_;
  msg_ = nullptr;
  msg_ = CopyMessage(msg);

  return *this;
}

char* Status::CopyMessage(const char* msg) {
  char* result = nullptr;
  if (msg != nullptr) {
    uint32_t size = ::strlen(msg) + 1;
    result = new char[size];
    ::memcpy(result, msg, size);
  }
  return result;
}

std::string Status::msg() const {
  if (msg_ == nullptr) {
    return "";
  }
  return msg_;
}

std::string Status::ToString() const {
  if (ok()) {
    return "OK";
  }

  char tmp[30];
  const char* type;
  switch (code_) {
    case error::CANCELLED:
      type = "Cancelled";
      break;
    case error::UNKNOWN:
      type = "Unknown";
      break;
    case error::INVALID_ARGUMENT:
      type = "Invalid argument";
      break;
    case error::DEADLINE_EXCEEDED:
      type = "Deadline exceeded";
      break;
    case error::NOT_FOUND:
      type = "Not found";
      break;
    case error::ALREADY_EXISTS:
      type = "Already exists";
      break;
    case error::PERMISSION_DENIED:
      type = "Permission denied";
      break;
    case error::UNAUTHENTICATED:
      type = "Unauthenticated";
      break;
    case error::RESOURCE_EXHAUSTED:
      type = "Resource exhausted";
      break;
    case error::FAILED_PRECONDITION:
      type = "Failed precondition";
      break;
    case error::ABORTED:
      type = "Aborted";
      break;
    case error::OUT_OF_RANGE:
      type = "Out of range";
      break;
    case error::UNIMPLEMENTED:
      type = "Unimplemented";
      break;
    case error::INTERNAL:
      type = "Internal";
      break;
    case error::UNAVAILABLE:
      type = "Unavailable";
      break;
    case error::DATA_LOSS:
      type = "Data loss";
      break;
    default:
      snprintf(tmp, sizeof(tmp), "Unknown code(%d)", static_cast<int>(code()));
      type = tmp;
      break;
  }

  std::string result(type);
  if (msg_ != nullptr) {
    result += ":";
    result.append(msg_);
  }
  return result;
}

}  // namespace deeprecmaster
