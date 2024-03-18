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

namespace deeprecmaster {
class PSResource {
 public:
  PSResource()
      : is_initialized_(false), has_ev_(false), ps_num_(0), ps_memory_(0) {}

  bool is_initialized() { return is_initialized_; }
  bool has_ev() { return has_ev_; }
  int ps_num() { return ps_num_; }
  int64_t ps_memory() { return ps_memory_; }

  void set_initialized() { is_initialized_ = true; }
  void set_has_ev(bool has_ev) { has_ev_ = has_ev; }
  void set_ps_num(int ps_num) { ps_num_ = ps_num; }
  void set_ps_memory(int64_t ps_memory) { ps_memory_ = ps_memory; }

 private:
  bool is_initialized_;
  bool has_ev_;
  int ps_num_;
  int64_t ps_memory_;
};

}  // namespace deeprecmaster
