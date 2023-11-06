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

#ifndef TF_FAULT_TOLERANCE_CC_KERNELS_SAVE_CACHE_CKPT_KERNELS_H_
#define TF_FAULT_TOLERANCE_CC_KERNELS_SAVE_CACHE_CKPT_KERNELS_H_

#include "tf_fault_tolerance/cc/kernels/cache_ckpt_kernels.h"

namespace tensorflow {

class GenerateCacheCKPTOp : public CacheCKPTOp {
 public:
  explicit GenerateCacheCKPTOp(OpKernelConstruction* context);
  ~GenerateCacheCKPTOp();

  void Compute(OpKernelContext* context) override;

 private:
  // Functions.
  // validate the inputs.
  void ValidateInputs(OpKernelContext* context, const Tensor& ckpt_path_prefix,
                      const Tensor& cache_path, const Tensor& shard,
                      const Tensor& num_shards);

  // Variables.
  bool is_merged_meta_;
  std::string ckpt_storage_type_;
};

class BackupRemoteCacheCKPTOp : public CacheCKPTOp {
 public:
  explicit BackupRemoteCacheCKPTOp(OpKernelConstruction* context);
  ~BackupRemoteCacheCKPTOp();

  void Compute(OpKernelContext* context) override;

 private:
  // Functions.
  // validate the inputs.
  void ValidateInputs(OpKernelContext* context, const Tensor& cache_path,
                      const Tensor& ckpt_key, const Tensor& ckpt_meta,
                      const Tensor& ckpt_data);

  // Variables.
  bool is_merged_meta_;
  std::string ckpt_storage_type_;
};

} // End of namespace tensorflow

#endif // End of TF_FAULT_TOLERANCE_CC_KERNEL_SAVE_CACHE_CKPT_KERNELS_H_
