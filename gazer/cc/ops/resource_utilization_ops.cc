#include <stdio.h>
#include <unistd.h>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {
namespace {
struct ProcMemory {
  long size;      // total program size
  long resident;  // resident set size
  long share;     // shared pages
  long trs;       // text (code)
  long lrs;       // library
  long drs;       // data/stack
  long dt;        // dirty pages
};

static long read_proc_memory() {
  ProcMemory m;
  errno = 0;
  FILE* fp = NULL;
  fp = fopen("/proc/self/statm", "r");
  if (NULL == fp) {
      return -1;
  }
  if (fscanf(fp, "%ld %ld %ld %ld %ld %ld %ld",
             &m.size, &m.resident, &m.share,
             &m.trs, &m.lrs, &m.drs, &m.dt) != 7) {

      return -1;
  }
  return m.resident * getpagesize();
}

} // namespace
namespace gazer {

REGISTER_OP("ResourceUtilization")
  .Output("cpu_usage: float32")
  .Output("memory_usage_in_mb: float32")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    c->set_output(0, c->Scalar());
    c->set_output(1, c->Scalar());
    return Status::OK();
  });

class ResourceUtilizationOp: public OpKernel {
 public:
  ResourceUtilizationOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    Tensor* cpu_usage = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {}, &cpu_usage));
    Tensor* mem_usage = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, {}, &mem_usage));

    auto cpu_usage_scalar = cpu_usage->scalar<float>();
    auto mem_usage_scalar = mem_usage->scalar<float>();
    // TODO compute cpu usage
    cpu_usage_scalar() = 100.0;
    mem_usage_scalar() = read_proc_memory() / 1024.0 / 1024.0;
    VLOG(1) << "MEM: " << mem_usage_scalar();
  }
};

REGISTER_KERNEL_BUILDER(Name("ResourceUtilization").Device(DEVICE_CPU),
                        ResourceUtilizationOp);

} // namespace gazer
} // namespace tensorflow
