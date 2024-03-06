#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
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

struct ProcStat {
  int pid;
  char name[100];
  char state;
  int ppid;
  int pgrp;
  int session;
  int tty_nr;
  int tpgid;
  int flags;

  long minflt;
  long cminflt;
  long majflt;
  long cmajflt;

  long utime;
  long stime;
  long cutime;
  long cstime;

  long priority;
  long nice;
  long num_threads;
  long itrealvalue;
  long starttime;
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

static long read_proc_cpu_time() {
  ProcStat s;
  errno = 0;
  FILE* fp = NULL;
  fp = fopen("/proc/self/stat", "r");
  if (NULL == fp) {
      return -1;
  }
  if (fscanf(fp, "%d %s %c %d %d "
                 "%d %d %d %d "
                 "%ld %ld %ld %ld "
                 "%ld %ld %ld %ld "
                 "%ld %ld %ld %ld %ld",
             &s.pid, s.name, &s.state, &s.ppid, &s.pgrp,
             &s.session, &s.tty_nr, &s.tpgid, &s.flags,
             &s.minflt, &s.cminflt, &s.majflt, &s.cmajflt,
             &s.utime, &s.stime, &s.cutime, &s.cstime,
             &s.priority, &s.nice, &s.num_threads,
             &s.itrealvalue, &s.starttime) != 22) {
      return -1;
  }
  return s.utime + s.stime;
}

int clockTicks = static_cast<int>(::sysconf(_SC_CLK_TCK));

} // namespace
namespace gazer {

REGISTER_OP("ResourceUtilization")
  .Output("cpu_usage: float32")
  .Output("memory_usage_in_mb: float32")
  // if not set stateful, this op will be optimize in constfold
  .SetIsStateful()
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    c->set_output(0, c->Scalar());
    c->set_output(1, c->Scalar());
    return Status::OK();
  });

class ResourceUtilizationOp: public OpKernel {
 public:
  ResourceUtilizationOp(OpKernelConstruction* ctx)
: OpKernel(ctx), last_cpu_time_(-1), last_time_stamp_(-1) {}

  void Compute(OpKernelContext* ctx) override {
    Tensor* cpu_usage = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {}, &cpu_usage));
    Tensor* mem_usage = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, {}, &mem_usage));

    auto cpu_usage_scalar = cpu_usage->scalar<float>();
    auto mem_usage_scalar = mem_usage->scalar<float>();
    // compute cpu usage
    int64 cpu_time = 0;
    float period = -1.0;
    if (-1 == last_cpu_time_) {
    } else {
      cpu_time = read_proc_cpu_time() - last_cpu_time_;
      period = (EnvTime::Default()->NowMicros() - last_time_stamp_) * 1.0
               / EnvTime::kSecondsToMicros;
    }
    last_cpu_time_ = read_proc_cpu_time();
    last_time_stamp_ = EnvTime::Default()->NowMicros();
    cpu_usage_scalar() = cpu_time / (period * clockTicks) * 100.0;
    // compute mem usage
    mem_usage_scalar() = read_proc_memory() / 1024.0 / 1024.0;
    VLOG(1) << "CPU: " << cpu_usage_scalar();
    VLOG(1) << "MEM: " << mem_usage_scalar();
  }
 private:
  int64 last_cpu_time_;
  int64 last_time_stamp_;
};

REGISTER_KERNEL_BUILDER(Name("ResourceUtilization").Device(DEVICE_CPU),
                        ResourceUtilizationOp);

} // namespace gazer
} // namespace tensorflow
