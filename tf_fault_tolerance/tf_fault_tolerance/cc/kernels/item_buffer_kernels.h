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

#ifndef TF_FAULT_TOLERANCE_CC_KERNELS_ITEM_BUFFER_KERNELS_H_
#define TF_FAULT_TOLERANCE_CC_KERNELS_ITEM_BUFFER_KERNELS_H_

#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

class ItemBuffer : public ResourceBase {
 public:
  explicit ItemBuffer(bool is_overwritable);
  ~ItemBuffer();

  Status Put(Tensor& record, int64 timeout_millis);
  Status Take(Tensor& record);
  Status SetState(bool is_cancelled);

#if (TF_MAJOR_VERSION * 1000L + TF_MINOR_VERSION) <= 1012L
  string DebugString() override;
#else
  string DebugString() const override;
#endif

  void Schedule(const string& name, int64 num_threads,
                std::function<void()> fn);
 private:
  bool is_overwritable_;
  std::mutex mu_;
  Tensor buffer_ GUARDED_BY(mu_);
  bool has_value_ GUARDED_BY(mu_);
  bool is_cancelled_ GUARDED_BY(mu_);
  std::condition_variable take_cv_;
  std::condition_variable put_cv_;
  std::shared_ptr<thread::ThreadPool> threads_ GUARDED_BY(mu_);
};

class ItemBufferOp : public OpKernel {
 public:
  explicit ItemBufferOp(OpKernelConstruction* ctx);

  void Compute(OpKernelContext* ctx) override;

 protected:
  virtual void ComputeWithItemBuffer(OpKernelContext* ctx,
                                     ItemBuffer* buf) = 0;

 private:
  bool overwritable_;
};

class ItemBufferAsyncOp : public AsyncOpKernel {
 public:
  explicit ItemBufferAsyncOp(OpKernelConstruction* ctx);

  void ComputeAsync(OpKernelContext* ctx,
                    AsyncOpKernel::DoneCallback done) override;

 private:
  // Functions.
  virtual void ComputeAsyncWithItemBuffer(OpKernelContext* ctx,
                                          AsyncOpKernel::DoneCallback done,
                                          ItemBuffer* buf) = 0;
  void Schedule(ItemBuffer* buffer, std::function<void()> fn);

  // Variables.
  string shared_name_;
  int64 shared_threads_;
  bool overwritable_;
};

class ItemBufferPutOp : public ItemBufferOp {
 public:
  explicit ItemBufferPutOp(OpKernelConstruction* ctx);

  void ComputeWithItemBuffer(OpKernelContext* ctx, ItemBuffer* buf) override;

 private:
  int64 timeout_millis_;
};

class ItemBufferTakeOp : public ItemBufferAsyncOp {
 public:
  explicit ItemBufferTakeOp(OpKernelConstruction* ctx);

  void ComputeAsyncWithItemBuffer(OpKernelContext* ctx,
                                  AsyncOpKernel::DoneCallback done,
                                  ItemBuffer* buf);
};

class ItemBufferSetStateOp : public ItemBufferOp {
 public:
  explicit ItemBufferSetStateOp(OpKernelConstruction* ctx);

  void ComputeWithItemBuffer(OpKernelContext* ctx, ItemBuffer* buf) override;

 private:
  bool is_cancelled_;
};

} // End of namespace tensorflow

#endif // End of TF_FAULT_TOLERANCE_CC_KERNELS_ITEM_BUFFER_KERNELS_H_
