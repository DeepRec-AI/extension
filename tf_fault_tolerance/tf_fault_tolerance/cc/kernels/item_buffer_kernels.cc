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

#include "tf_fault_tolerance/cc/kernels/item_buffer_kernels.h"

namespace tensorflow {

//------------------------------------------------------------------------------
// Functions of ItemBuffer.

ItemBuffer::ItemBuffer(bool is_overwritable)
  : is_overwritable_(is_overwritable), has_value_(false),
    is_cancelled_(false) {}

ItemBuffer::~ItemBuffer() { SetState(true); }

Status ItemBuffer::Put(Tensor& record, int64 timeout_millis) {
  std::unique_lock<std::mutex> lock(mu_);

  bool should_retry = \
    !put_cv_.wait_for(lock, std::chrono::milliseconds(timeout_millis),
    [this]() {
      return is_cancelled_ || is_overwritable_ || has_value_ == false;
    });
  if (should_retry) {
    lock.unlock();
    LOG(WARNING) << "ItemBuffer Put was ignored since timeout.";
    return Status::OK();
  }

  if (TF_PREDICT_FALSE(is_cancelled_)) {
    lock.unlock();
    return Status(errors::Cancelled("Session was closed."));
  }

  if (TF_PREDICT_FALSE(has_value_)) {
    LOG(WARNING) << "ItemBuffer Put will overwrite old item. Please check "
                 << "the Put interval.";
  }

  buffer_ = record;
  has_value_ = true;

  lock.unlock();
  take_cv_.notify_all();

  return Status::OK();
}

Status ItemBuffer::Take(Tensor& record) {
  std::unique_lock<std::mutex> lock(mu_);

  take_cv_.wait(lock, [this]() { return is_cancelled_ || has_value_; });

  if (TF_PREDICT_FALSE(is_cancelled_ && has_value_ == false)) {
    lock.unlock();
    return Status(errors::Cancelled("Session was closed."));
  }

  record = buffer_;
  has_value_ = false;

  lock.unlock();
  put_cv_.notify_all();

  return Status::OK();
}

Status ItemBuffer::SetState(bool is_cancelled) {
  std::unique_lock<std::mutex> lock(mu_);

  is_cancelled_ = is_cancelled;

  lock.unlock();
  put_cv_.notify_all();
  take_cv_.notify_all();

  return Status::OK();
}

string ItemBuffer::DebugString() const {
  return strings::StrCat("ItemBuffer(has_value=", has_value_, "item=",
                         buffer_.DeviceSafeDebugString());
}

void ItemBuffer::Schedule(const string& name, int64 num_threads,
                          std::function<void()> fn) {
  std::unique_lock<std::mutex> lock(mu_);
  if (threads_) {
    lock.unlock();
    threads_->Schedule(fn);
    return;
  }

  threads_.reset(
    new thread::ThreadPool(Env::Default(), ThreadOptions(),
                           strings::StrCat("item_buffer_threads_", name),
                           num_threads, /* low_latency_hint= */false));

  lock.unlock();
  threads_->Schedule(fn);
}

//------------------------------------------------------------------------------
// Functions of ItemBufferOp.

ItemBufferOp::ItemBufferOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

void ItemBufferOp::Compute(OpKernelContext* ctx) {
  auto rm = ctx->resource_manager();
  auto ndef = def();

  ContainerInfo cinfo;
  OP_REQUIRES_OK(ctx, cinfo.Init(rm, ndef, /* use name */true));

  ItemBuffer* buffer = nullptr;
  OP_REQUIRES_OK(ctx, rm->LookupOrCreate<ItemBuffer>(
                            cinfo.container(), cinfo.name(), &buffer,
                            [&ndef](ItemBuffer** pbuf) -> Status {
                              bool is_overwritable = false;
                              TF_RETURN_IF_ERROR(GetNodeAttr(
                                ndef, "is_overwritable", &is_overwritable));
                              *pbuf = new ItemBuffer(is_overwritable);
                              return Status::OK();
                            }));
  core::ScopedUnref scope(buffer);
  ComputeWithItemBuffer(ctx, buffer);
}

//------------------------------------------------------------------------------
// Functions of ItemBufferAsyncOp.

ItemBufferAsyncOp::ItemBufferAsyncOp(OpKernelConstruction* ctx)
  : AsyncOpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr("shared_name", &shared_name_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("shared_threads", &shared_threads_));
}

void ItemBufferAsyncOp::ComputeAsync(OpKernelContext* ctx,
                                     AsyncOpKernel::DoneCallback done) {
  auto rm = ctx->resource_manager();
  auto ndef = def();

  ContainerInfo cinfo;
  OP_REQUIRES_OK_ASYNC(ctx, cinfo.Init(rm, ndef, /* use name */true), done);

  ItemBuffer* buffer = nullptr;
  OP_REQUIRES_OK_ASYNC(ctx, rm->LookupOrCreate<ItemBuffer>(
                                  cinfo.container(), cinfo.name(), &buffer,
                                  [&ndef](ItemBuffer** pbuf) -> Status {
                                    bool is_overwritable = false;
                                    TF_RETURN_IF_ERROR(GetNodeAttr(ndef,
                                      "is_overwritable", &is_overwritable));
                                    *pbuf = new ItemBuffer(is_overwritable);
                                    return Status::OK();
                                  }), done);
  core::ScopedUnref scope(buffer);
  Schedule(buffer, [this, ctx, done, buffer](){
    ComputeAsyncWithItemBuffer(ctx, done, buffer);
  });
}

void ItemBufferAsyncOp::Schedule(ItemBuffer* buffer, std::function<void()> fn) {
  buffer->Schedule(shared_name_, shared_threads_, fn);
}

//------------------------------------------------------------------------------
// Functions of ItemBufferPutOp.

ItemBufferPutOp::ItemBufferPutOp(OpKernelConstruction* ctx)
  : ItemBufferOp(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr("timeout_millis", &timeout_millis_));
}

void ItemBufferPutOp::ComputeWithItemBuffer(OpKernelContext* ctx,
                                            ItemBuffer* buf) {
  Tensor record_t = ctx->input(0);
  OP_REQUIRES_OK(ctx, buf->Put(record_t, timeout_millis_));
}

REGISTER_KERNEL_BUILDER(Name("ItemBufferPut").Device(DEVICE_CPU),
                        ItemBufferPutOp);

//------------------------------------------------------------------------------
// Functions of ItemBufferTakeOp.

ItemBufferTakeOp::ItemBufferTakeOp(OpKernelConstruction* ctx)
  : ItemBufferAsyncOp(ctx) {}

void ItemBufferTakeOp::ComputeAsyncWithItemBuffer(OpKernelContext* ctx,
                         AsyncOpKernel::DoneCallback done, ItemBuffer* buf) {
  Tensor* record_t = nullptr;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &record_t));

  Status s = buf->Take(*record_t);
  if (TF_PREDICT_FALSE(!s.ok())) {
    ctx->SetStatus(s);
    done();
    return;
  }

  done();
}

REGISTER_KERNEL_BUILDER(Name("ItemBufferTake").Device(DEVICE_CPU),
                        ItemBufferTakeOp);

//------------------------------------------------------------------------------
// Functions of ItemBufferSetStateOp.

ItemBufferSetStateOp::ItemBufferSetStateOp(OpKernelConstruction* ctx)
  : ItemBufferOp(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr("is_cancelled", &is_cancelled_));
}

void ItemBufferSetStateOp::ComputeWithItemBuffer(OpKernelContext* ctx,
                                                 ItemBuffer* buf) {
  OP_REQUIRES_OK(ctx, buf->SetState(is_cancelled_));
}

REGISTER_KERNEL_BUILDER(Name("ItemBufferSetState").Device(DEVICE_CPU),
                        ItemBufferSetStateOp);

} // End of namespace tensorflow
