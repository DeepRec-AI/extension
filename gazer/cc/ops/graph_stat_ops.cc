#include <stdio.h>
#include <unistd.h>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/env_var.h"

#if (TF_MAJOR_VERSION * 1000L + TF_MINOR_VERSION) >= 1015L
#include "tensorflow/core/framework/embedding/embedding_var.h"
#elif (TF_MAJOR_VERSION * 1000L + TF_MINOR_VERSION) >= 1012L
#include "tensorflow/core/framework/embedding_var.h"
#else // version = 1.4.0
#include "tensorflow/core/kernels/kv_variable_ops.h"
#endif

namespace tensorflow {
namespace {
const char* kItemSeparator = "@";
const char* kKVSeparator = ":";
const char* kGlobalResourceName = "GLOBAL_RESOURCE_NAME";
} // namespace
namespace gazer {

REGISTER_OP("ResourceStat")
  .Output("memory_usage_in_mb: float32")
  .SetIsStateful()
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    c->set_output(0, c->Scalar());
    return Status::OK();
  });

class ResourceStatOp: public OpKernel {
 public:
  ResourceStatOp(OpKernelConstruction* ctx)
: OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    std::string names;
    ReadStringFromEnvVar(kGlobalResourceName, "", &names);
    std::vector<std::string> resource_names = str_util::Split(names, kItemSeparator);

    ResourceMgr* rm = ctx->resource_manager();
    int64 total_size = 0;
    for (const auto & resource_name : resource_names) {
      std::vector<std::string> item = str_util::Split(resource_name, kKVSeparator);
      if (item.size() == 2) {
        int64 var_size = std::stoi(item[1]);
        if (var_size == -1) {
          EmbeddingVar<int64, float>* r = nullptr;
          TF_CHECK_OK(rm->Lookup("", item[0], &r));
          // TODO add EV size.
          LOG(INFO) << "ev: " << r->DebugString();
          r->Unref();
        } else {
          total_size += var_size;
        }
      }
    }

    Tensor* mem_usage = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {}, &mem_usage));
    mem_usage->scalar<float>()() = total_size / 1024.0 / 1024.0;
    VLOG(1) << "resource MEM: " << mem_usage->scalar<float>()();
  }
 private:
};

REGISTER_KERNEL_BUILDER(Name("ResourceStat").Device(DEVICE_CPU),
                        ResourceStatOp);

REGISTER_OP("TimeStamp")
  .Output("time_stamp_in_us: int64")
  .SetIsStateful()
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    c->set_output(0, c->Scalar());
    return Status::OK();
  });

class TimeStampOp: public OpKernel {
 public:
  TimeStampOp(OpKernelConstruction* ctx)
: OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    Tensor* time_stamp = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {}, &time_stamp));
    time_stamp->scalar<int64>()() = EnvTime::Default()->NowMicros();
  }
};

REGISTER_KERNEL_BUILDER(Name("TimeStamp").Device(DEVICE_CPU),
                        TimeStampOp);

} // namespace gazer
} // namespace tensorflow
