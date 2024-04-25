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

#include <chrono>
#include <vector>
#include <queue>

namespace deeprecmaster {

template <class T>
class GlobalWorkQueue {
 public:
  GlobalWorkQueue(const std::vector<T>& works)
      : GlobalWorkQueue(works, 1, false) {}

  GlobalWorkQueue(const std::vector<T>& works, int32_t num_epochs)
      : GlobalWorkQueue(works, num_epochs, false) {}

  GlobalWorkQueue(
      const std::vector<T>& works, int32_t num_epochs, bool shuffle)
      : shuffle_(shuffle), num_epochs_(num_epochs),
        cur_epochs_(0), index_in_epoch(0), src_works_(works) {
    seeds_.resize(num_epochs_+1, 0);
    for (int32_t i=0; i<num_epochs; ++i) {
      FillWorkQueue(0, i, src_works_);
    }
  }

  GlobalWorkQueue(
      const std::vector<T>& works, int32_t num_epochs, bool shuffle,
      int64_t seed, int32_t cur_epochs, int32_t drop_num=0)
      : shuffle_(shuffle), num_epochs_(num_epochs),
        cur_epochs_(cur_epochs), index_in_epoch(0), src_works_(works) {
    seeds_.resize(num_epochs_+1, 0);
    FillWorkQueue(seed, cur_epochs, src_works_);

    for (int32_t i=cur_epochs+1; i<num_epochs; ++i) {
      FillWorkQueue(0, i, src_works_);
    }

    Pop(drop_num);
  }

  bool Front(T& elem) {
    if (IsEmpty()) {
      return false;
    }
    elem = work_queue_.front();
    return true;
  }

  bool Back(T& elem) {
    if (IsEmpty()) {
      return false;
    }
    elem = work_queue_.back();
    return true;
  }

  void Put(const T& elem) {
    work_queue_.push(elem);
  }

  void Pop(int64_t size=1) {
    while (!work_queue_.empty() && size-- != 0) {
      work_queue_.pop();
      index_in_epoch += 1;
      if (index_in_epoch >= src_works_.size()) {
        index_in_epoch = 0;
        cur_epochs_ += 1;
      }
    }
  }

  int64_t Size() {
    return work_queue_.size();
  }

  bool IsShuffle() {
    return shuffle_;
  }

  bool IsEmpty() {
    return work_queue_.empty();
  }

  int32_t GetCurrentEpoch() {
    return cur_epochs_;
  }

  int32_t GetCurrentSliceIndex() {
    return index_in_epoch;
  }

  int64_t GetCurrentShuffleSeed() {
    if (cur_epochs_ >= num_epochs_) {
      return 0;
    } else {
      return seeds_.at(cur_epochs_);
    }
  }

 private:
  void FillWorkQueue(int64_t seed, int32_t cur_epochs, std::vector<T> works) {
    if (cur_epochs >= num_epochs_) {
      return;
    }
    if (shuffle_) {
      seed = (seed == 0) ? GetShuffleSeed() : seed;
      std::default_random_engine dre(seed);
      std::shuffle(works.begin(), works.end(), dre);
    }
    seeds_[cur_epochs] = seed;
    for (const auto& work : works) {
      work_queue_.push(work);
    }
  }

  int64_t GetShuffleSeed(int64_t latency=10) {
    usleep(latency);
    std::chrono::time_point<std::chrono::system_clock,
                            std::chrono::microseconds>
        tp = std::chrono::time_point_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now());
    auto tmp = std::chrono::duration_cast<std::chrono::microseconds>(
                  tp.time_since_epoch());
    return tmp.count();
  }

 private:
  bool shuffle_;
  int32_t num_epochs_;
  int32_t cur_epochs_;
  int32_t index_in_epoch;
  std::queue<T> work_queue_;
  std::vector<int64_t> seeds_;
  std::vector<T> src_works_;
};

} // End of namespace deeprecmaster
