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

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

class GenerateCacheCKPTOp : public OpKernel {
 public:
  explicit GenerateCacheCKPTOp(OpKernelConstruction* context);

  ~GenerateCacheCKPTOp();

  void Compute(OpKernelContext* context) override;

 private:
  // validate the inputs.
  void ValidateInputs(OpKernelContext* context, const Tensor& ckpt_prefix,
                      const Tensor& cache_ckpt_path, const Tensor& shard,
                      const Tensor& num_shards);

  // create local cache checkpoint file.
  Status CreateLocalCacheCKPT(const std::string& ckpt_path_prefix,
                              const std::string& cache_ckpt_path,
                              int32 shard_id, int32 num_shards,
                              std::string& cache_ckpt_data_file);

  // generate cache_ckpt_key output, RecvRemoteCacheCKPTOp will use it to create
  // a mapping to cache ckpt.
  Status FillCacheCKPTKeyTensor(const std::string& ckpt_path_prefix,
                                int32 shard_id, int32 num_shards, Tensor* t);

  // generate cache_ckpt, which will be sent to other ps.
  Status FillCacheCKPTTensor(const std::string& cache_ckpt_data_file,
                             Tensor* t);
};

class RecvRemoteCacheCKPTOp : public OpKernel {
 public:
  explicit RecvRemoteCacheCKPTOp(OpKernelConstruction* context);

  ~RecvRemoteCacheCKPTOp();

  void Compute(OpKernelContext* context) override;
};

} // End of namespace tensorflow

#endif // End of TF_FAULT_TOLERANCE_CC_KERNEL_SAVE_CACHE_CKPT_KERNELS_H_
