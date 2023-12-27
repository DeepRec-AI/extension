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

#ifndef TF_FAULT_TOLERANCE_CC_KERNELS_RESTORE_CACHE_CKPT_KERNELS_H_
#define TF_FAULT_TOLERANCE_CC_KERNELS_RESTORE_CACHE_CKPT_KERNELS_H_

#include "tf_fault_tolerance/cc/kernels/cache_ckpt_kernels.h"

namespace tensorflow {

class CheckLocalCacheCKPTOp : public CacheCKPTOp {
 public:
  explicit CheckLocalCacheCKPTOp(OpKernelConstruction* context);
  ~CheckLocalCacheCKPTOp();

  void Compute(OpKernelContext* context) override;

 private:
  // Function.
  void ValidateInputs(OpKernelContext* context, const Tensor& ckpt_prefix,
                      const Tensor& shard, const Tensor& num_shards);

  Status TryToGetCacheCKPT(OpKernelContext* ctx, const std::string& ckpt_key,
                           ResourceHandle& handle, bool& is_exist_cache);

  // Variable.
  std::string container_;
  std::string name_;
};

class GetRemoteCacheCKPTOp : public CacheCKPTOp {
 public:
  explicit GetRemoteCacheCKPTOp(OpKernelConstruction* context);
  ~GetRemoteCacheCKPTOp();

  void Compute(OpKernelContext* context) override;

 private:
  void ValidateInputs(OpKernelContext* context, const Tensor& ckpt_path_prefix,
                      const Tensor& cache_path, const Tensor& exist_cache_ckpt,
                      const Tensor& ckpt_key, const Tensor& ckpt_meta,
                      const Tensor& ckpt_data);

  Status GenerateLocalCacheCKPT(OpKernelContext* context,
                                ResourceHandle& handle,
                                const std::string& ckpt_key,
                                const std::string& cache_path,
                                const std::string& ckpt_filename_prefix,
                                const std::string& meta_file,
                                const std::string& data_file,
                                bool& exist_cache_ckpt);

  // Variable.
  std::string container_;
  std::string name_;
  bool is_merged_meta_;
  std::string ckpt_storage_type_;
};

class RepatriateRemoteCacheCKPTOp : public CacheCKPTOp {
 public:
  explicit RepatriateRemoteCacheCKPTOp(OpKernelConstruction* context);
  ~RepatriateRemoteCacheCKPTOp();

  void Compute(OpKernelContext* context) override;

 private:
  // Functions.
  void ValidateInputs(OpKernelContext* context, const Tensor& ckpt_path_prefix,
                      const Tensor& shard, const Tensor& num_shards);

  bool FillCacheCKPTOutput(const std::string& ckpt_key, Tensor& ckpt_meta,
                           Tensor& ckpt_data);

  // Variables.
  bool output_is_path_;
};

class LoadCKPTFromFilePathOp : public CacheCKPTOp {
 public:
  explicit LoadCKPTFromFilePathOp(OpKernelConstruction* context);
  ~LoadCKPTFromFilePathOp();

  void Compute(OpKernelContext* context) override;

 private:
  // Functions.
  void ValidateInputs(OpKernelContext* context, const Tensor& ckpt_path_prefix,
                      const Tensor& shard_path, const Tensor& num_shards);
  // Variables.
  std::string container_;
  std::string name_;
  bool output_is_path_;
};

class UnPackCacheCKPTResourceOp : public CacheCKPTOp{
 public:
  explicit UnPackCacheCKPTResourceOp(OpKernelConstruction* context);
  ~UnPackCacheCKPTResourceOp();

  void Compute(OpKernelContext* context) override;
};

} // End of namespace tensorflow

#endif // End of TF_FAULT_TOLERANCE_CC_KERNELS_RESTORE_CACHE_CKPT_KERNELS_H_
