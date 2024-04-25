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

#include <dirent.h>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>

#include "deeprec_master/include/errors.h"
#include "deeprec_master/include/file.h"

namespace deeprecmaster {

namespace {

std::string TranslateName(const std::string& name) {
  // If the name is empty, FormatPath returns "." which is incorrect and
  // we should return the empty path instead.
  if (name.empty()) return name;
  return FormatPath(name);
}

error::Code ErrnoToCode(int err_number) {
  error::Code code;
  switch (err_number) {
    case 0:
      code = error::OK;
      break;
    case EINVAL:        // Invalid argument
    case ENAMETOOLONG:  // Filename too long
    case E2BIG:         // Argument list too long
    case EDESTADDRREQ:  // Destination address required
    case EDOM:          // Mathematics argument out of domain of function
    case EFAULT:        // Bad address
    case EILSEQ:        // Illegal byte sequence
    case ENOPROTOOPT:   // Protocol not available
    case ENOSTR:        // Not a STREAM
    case ENOTSOCK:      // Not a socket
    case ENOTTY:        // Inappropriate I/O control operation
    case EPROTOTYPE:    // Protocol wrong type for socket
    case ESPIPE:        // Invalid seek
      code = error::INVALID_ARGUMENT;
      break;
    case ETIMEDOUT:  // Connection timed out
    case ETIME:      // Timer expired
      code = error::DEADLINE_EXCEEDED;
      break;
    case ENODEV:  // No such device
    case ENOENT:  // No such file or directory
    case ENXIO:   // No such device or address
    case ESRCH:   // No such process
      code = error::NOT_FOUND;
      break;
    case EEXIST:         // File exists
    case EADDRNOTAVAIL:  // Address not available
    case EALREADY:       // Connection already in progress
      code = error::ALREADY_EXISTS;
      break;
    case EPERM:   // Operation not permitted
    case EACCES:  // Permission denied
    case EROFS:   // Read only file system
      code = error::PERMISSION_DENIED;
      break;
    case ENOTEMPTY:   // Directory not empty
    case EISDIR:      // Is a directory
    case ENOTDIR:     // Not a directory
    case EADDRINUSE:  // Address already in use
    case EBADF:       // Invalid file descriptor
    case EBUSY:       // Device or resource busy
    case ECHILD:      // No child processes
    case EISCONN:     // Socket is connected
    case ENOTBLK:  // Block device required
    case ENOTCONN:  // The socket is not connected
    case EPIPE:     // Broken pipe
    case ESHUTDOWN:  // Cannot send after transport endpoint shutdown
    case ETXTBSY:  // Text file busy
      code = error::FAILED_PRECONDITION;
      break;
    case ENOSPC:  // No space left on device
    case EDQUOT:  // Disk quota exceeded
    case EMFILE:   // Too many open files
    case EMLINK:   // Too many links
    case ENFILE:   // Too many open files in system
    case ENOBUFS:  // No buffer space available
    case ENODATA:  // No message is available on the STREAM read queue
    case ENOMEM:   // Not enough space
    case ENOSR:    // No STREAM resources
    case EUSERS:  // Too many users
      code = error::RESOURCE_EXHAUSTED;
      break;
    case EFBIG:      // File too large
    case EOVERFLOW:  // Value too large to be stored in data type
    case ERANGE:     // Result too large
      code = error::OUT_OF_RANGE;
      break;
    case ENOSYS:        // Function not implemented
    case ENOTSUP:       // Operation not supported
    case EAFNOSUPPORT:  // Address family not supported
    case EPFNOSUPPORT:  // Protocol family not supported
    case EPROTONOSUPPORT:  // Protocol not supported
    case ESOCKTNOSUPPORT:  // Socket type not supported
    case EXDEV:  // Improper link
      code = error::UNIMPLEMENTED;
      break;
    case EAGAIN:        // Resource temporarily unavailable
    case ECONNREFUSED:  // Connection refused
    case ECONNABORTED:  // Connection aborted
    case ECONNRESET:    // Connection reset
    case EINTR:         // Interrupted function call
    case EHOSTDOWN:  // Host is down
    case EHOSTUNREACH:  // Host is unreachable
    case ENETDOWN:      // Network is down
    case ENETRESET:     // Connection aborted by network
    case ENETUNREACH:   // Network unreachable
    case ENOLCK:        // No locks available
    case ENOLINK:       // Link has been severed
    case ENONET:  // Machine is not on the network
      code = error::UNAVAILABLE;
      break;
    case EDEADLK:  // Resource deadlock avoided
    case ESTALE:  // Stale file handle
      code = error::ABORTED;
      break;
    case ECANCELED:  // Operation cancelled
      code = error::CANCELLED;
      break;
    // NOTE: If you get any of the following (especially in a
    // reproducible way) and can propose a better mapping,
    // please email the owners about updating this mapping.
    case EBADMSG:      // Bad message
    case EIDRM:        // Identifier removed
    case EINPROGRESS:  // Operation in progress
    case EIO:          // I/O error
    case ELOOP:        // Too many levels of symbolic links
    case ENOEXEC:      // Exec format error
    case ENOMSG:       // No message of the desired type
    case EPROTO:       // Protocol error
    case EREMOTE:  // Object is remote
      code = error::UNKNOWN;
      break;
    default: {
      code = error::UNKNOWN;
      break;
    }
  }
  return code;
}

Status IOError(const std::string& context, int err_number) {
  auto code = ErrnoToCode(err_number);
  std::ostringstream emsg;
  emsg << context << "; " << err_number;
  return Status(code, emsg.str());
}

} // End of anonymous namespace

std::string FormatPath(const std::string& unclean_path) {
  std::string path(unclean_path);
  const char* src = path.c_str();
  std::string::iterator dst = path.begin();

  // Check for absolute path and determine initial backtrack limit.
  const bool is_absolute_path = *src == '/';
  if (is_absolute_path) {
    *dst++ = *src++;
    while (*src == '/') ++src;
  }
  std::string::const_iterator backtrack_limit = dst;

  // Process all parts
  while (*src) {
    bool parsed = false;

    if (src[0] == '.') {
      //  1dot ".<whateverisnext>", check for END or SEP.
      if (src[1] == '/' || !src[1]) {
        if (*++src) {
          ++src;
        }
        parsed = true;
      } else if (src[1] == '.' && (src[2] == '/' || !src[2])) {
        // 2dot END or SEP (".." | "../<whateverisnext>").
        src += 2;
        if (dst != backtrack_limit) {
          // We can backtrack the previous part
          for (--dst; dst != backtrack_limit && dst[-1] != '/'; --dst) {
            // Empty.
          }
        } else if (!is_absolute_path) {
          // Failed to backtrack and we can't skip it either. Rewind and copy.
          src -= 2;
          *dst++ = *src++;
          *dst++ = *src++;
          if (*src) {
            *dst++ = *src;
          }
          // We can never backtrack over a copied "../" part so set new limit.
          backtrack_limit = dst;
        }
        if (*src) {
          ++src;
        }
        parsed = true;
      }
    }

    // If not parsed, copy entire part until the next SEP or EOS.
    if (!parsed) {
      while (*src && *src != '/') {
        *dst++ = *src++;
      }
      if (*src) {
        *dst++ = *src++;
      }
    }

    // Skip consecutive SEP occurrences
    while (*src == '/') {
      ++src;
    }
  }

  // Calculate and check the length of the cleaned path.
  std::string::difference_type path_length = dst - path.begin();
  if (path_length != 0) {
    // Remove trailing '/' except if it is root path ("/" ==> path_length := 1)
    if (path_length > 1 && path[path_length - 1] == '/') {
      --path_length;
    }
    path.resize(path_length);
  } else {
    // The cleaned path is empty; assign "." as per the spec.
    path.assign(1, '.');
  }
  return path;
}

Status FileExists(const std::string& fname) {
  if (access(TranslateName(fname).c_str(), F_OK) == 0) {
    return Status::OK();
  }
  return error::NotFound(fname + " not found");
}

Status Stat(const std::string& fname, FileStatistics& stats) {
  Status s;
  struct stat sbuf;
  if (stat(TranslateName(fname).c_str(), &sbuf) != 0) {
    s = IOError(fname, errno);
  } else {
    stats.length = sbuf.st_size;
    stats.mtime_nsec = sbuf.st_mtime * 1e9;
    stats.is_directory = S_ISDIR(sbuf.st_mode);
  }
  return s;
}

Status IsDirectory(const std::string& name) {
  // Check if path exists.
  RETURN_IF_ERROR(FileExists(name));
  FileStatistics stat;
  RETURN_IF_ERROR(Stat(name, stat));
  if (stat.is_directory) {
    return Status::OK();
  }
  return Status(error::FAILED_PRECONDITION, "Not a directory");
}

Status GetChildren(const std::string& dir, std::vector<std::string>& result) {
  std::string translated_dir = TranslateName(dir);
  result.clear();
  DIR* d = opendir(translated_dir.c_str());
  if (d == nullptr) {
    return IOError(dir, errno);
  }
  struct dirent* entry;
  while ((entry = readdir(d)) != nullptr) {
    std::string basename = entry->d_name;
    if ((basename != ".") && (basename != "..")) {
      result.push_back(entry->d_name);
    }
  }
  closedir(d);
  return Status::OK();
}

} // End of namespace deeprecmaster
