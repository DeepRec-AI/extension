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

#ifndef TF_FAULT_TOLERANCE_CC_KERNELS_CACHE_CKPT_KERNELS_H_
#define TF_FAULT_TOLERANCE_CC_KERNELS_CACHE_CKPT_KERNELS_H_

#include <string>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/public/version.h"

#include "tf_fault_tolerance/cc/common/cache_ckpt/cache_ckpt_manager.h"

namespace tensorflow {

class CacheCKPTVar : public ResourceBase {
 public:
  CacheCKPTVar();
  ~CacheCKPTVar();

  mutex* mu();
  void update(const Tensor& meta_tensor, const Tensor& data_tensor);
  Tensor* meta_tensor();
  Tensor* data_tensor();

#if (TF_MAJOR_VERSION * 1000L + TF_MINOR_VERSION) <= 1012L
  std::string DebugString() override;
#else
  std::string DebugString() const override;
#endif

  TF_DISALLOW_COPY_AND_ASSIGN(CacheCKPTVar);

 private:
  mutex mu_;
  Tensor meta_tensor_;
  Tensor data_tensor_;
};

class CacheCKPTOp : public OpKernel {
 public:
  explicit CacheCKPTOp(OpKernelConstruction* context);
  ~CacheCKPTOp();

 protected:
  // Get CacheCKPTManager, if not exist, create a new one.
  Status GetOrCreateCacheCKPTVar(OpKernelContext* context,
                                 const ResourceHandle& handle,
                                 CacheCKPTVar*& cache_ckpt);
};

} // End of namespace tensorflow.

#endif // End of TF_FAULT_TOLERANCE_CC_KERNELS_CACHE_CKPT_KERNELS_H_
