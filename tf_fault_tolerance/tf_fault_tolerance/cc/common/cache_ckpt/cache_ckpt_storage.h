/* Copyright 2023 The DeepRec Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TF_FAULT_TOLERANCE_CC_COMMON_CACHE_CKPT_CACHE_CKPT_STORAGE_H_
#define TF_FAULT_TOLERANCE_CC_COMMON_CACHE_CKPT_CACHE_CKPT_STORAGE_H_

#include <string>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"


namespace tensorflow {

class CacheCKPTStorage {
 public:
  CacheCKPTStorage(const int64 global_step);
  virtual ~CacheCKPTStorage();

  virtual Status Write(const Tensor& ckpt_meta_tenosr,
                       const Tensor& ckpt_data_tensor) = 0;

  virtual Status Read(Tensor& ckpt_meta_tensor,
                      Tensor& ckpt_data_tensor) const = 0;

  int64 global_step() const;

 private:
  int64 global_step_;
};

class MemoryCacheCKPTStorage : public CacheCKPTStorage {
 public:
  MemoryCacheCKPTStorage(const int64 global_step);

  ~MemoryCacheCKPTStorage();

  Status Write(const Tensor& ckpt_meta_tensor,
               const Tensor& ckpt_data_tensor) override;

  Status Read(Tensor& ckpt_meta_tensor,
              Tensor& ckpt_data_tensor) const override;

 private:
  Tensor cache_ckpt_meta_tensor_;
  Tensor cache_ckpt_data_tensor_;
};

class POSIXFileCacheCKPTStorage : public CacheCKPTStorage {
 public:
  /**
   * @brief Create POSIXFileCacheCKPTStorage.
   *
   * @param [in] cache_path: path to save cache ckpt file.
   * @param [in] ckpt_filename_prefix: filename prefix without global_step,
   *               for example: 'model.ckpt'
   * @param [in] global_step: global_step of this ckpt.
   * @param [in] shard: shard id of this ckpt.
   * @param [in] num_shards: num_shards of this ckpt.
   * @param [in] is_local_ckpt: this ckpt is local ckpt or not.
   * @param [in] is_merge_meta: meta file is merged or not.
   */
  POSIXFileCacheCKPTStorage(const std::string& cache_path,
                            const std::string& ckpt_filename_prefix,
                            const int64 global_step, const int32 shard,
                            const int32 num_shards, const bool is_local_ckpt,
                            const bool is_merged_meta);

  ~POSIXFileCacheCKPTStorage();

  Status Write(const Tensor& ckpt_meta_tensor,
               const Tensor& ckpt_data_tensor) override;
  Status Read(Tensor& ckpt_meta_tensor,
              Tensor& ckpt_data_tensor) const override;

 private:
  // Variables.
  std::string cache_path_;
  std::string cache_ckpt_file_path_prefix_;
  std::string cache_ckpt_meta_file_path_;
  std::string cache_ckpt_data_file_path_;
  bool is_local_ckpt_;
};

} // End of namespace tensorflow

#endif // End of TF_FAULT_TOLERANCE_CC_COMMON_CACHE_CKPT_CACHE_CKPT_STORAGE_H_
