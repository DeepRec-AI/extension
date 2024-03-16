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

#include "tensorflow/core/lib/strings/stringprintf.h"

#include "tf_fault_tolerance/cc/kernels/cache_ckpt_kernels.h"

namespace tensorflow {

//------------------------------------------------------------------------------
// functions of CacheCKPT.
CacheCKPTVar::CacheCKPTVar() {}

CacheCKPTVar::~CacheCKPTVar() {}

mutex* CacheCKPTVar::mu() {
  return &mu_;
}

void CacheCKPTVar::update(const Tensor& meta_tensor,
                          const Tensor& data_tensor) {
  meta_tensor_ = meta_tensor;
  data_tensor_ = data_tensor;
}

Tensor* CacheCKPTVar::meta_tensor() {
  return &meta_tensor_;
}

Tensor* CacheCKPTVar::data_tensor() {
  return &data_tensor_;
}

#if (TF_MAJOR_VERSION * 1000L + TF_MINOR_VERSION) <= 1012L
std::string CacheCKPTVar::DebugString() {
#else
std::string CacheCKPTVar::DebugString() const {
#endif

  std::string val = \
    strings::Printf("meta_ckpt(size: %ld bytes), data_ckpt(size: %ld bytes)",
                    meta_tensor_.TotalBytes(), data_tensor_.TotalBytes());
  return val;
}

//------------------------------------------------------------------------------
// functions of CacheCKPTOp.
CacheCKPTOp::CacheCKPTOp(OpKernelConstruction* context)
  : OpKernel(context) {}

CacheCKPTOp::~CacheCKPTOp() {}

Status CacheCKPTOp::GetOrCreateCacheCKPTVar(OpKernelContext* context,
                      const ResourceHandle& handle, CacheCKPTVar*& cache_ckpt) {

  TF_RETURN_IF_ERROR(LookupOrCreateResource<CacheCKPTVar>(context, handle,
                       &cache_ckpt, [](CacheCKPTVar** p) -> Status {
                         *p = new CacheCKPTVar();
                         return Status::OK();
                       }));

  return Status::OK();
}

} // End of namespace tensorflow.
