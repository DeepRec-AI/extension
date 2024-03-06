#include <stdio.h>
#include <unistd.h>

#include <memory>
#include <unordered_set>

#include "gazer/cc/client/grpc_client.h"
#include "rapidjson/document.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
using shape_inference::InferenceContext;
namespace gazer {

REGISTER_OP("ReportAIMaster")
    .Input("input: N * string")
    .Output("output: N * string")
    .Attr("N: int >= 1")
    .Attr("job: string = ''")
    .Attr("task_id: int >= 0")
    .SetShapeFn([](InferenceContext* ctx) {
      int num_lookups = ctx->num_outputs();
      for (int i = 0; i < num_lookups; ++i) {
        ctx->set_output(i, ctx->Scalar());
      }
      return Status::OK();
    });

class ReportAIMasterOp : public OpKernel {
 public:
  explicit ReportAIMasterOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("N", &num_summary_inputs));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("job", &job_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("task_id", &task_id_));
  }
  void Compute(OpKernelContext* ctx) override {
    std::unordered_set<std::string> tags;
    std::vector<std::string> metric_types;
    std::vector<float> metric_values;
    // Summary merged_summary;
    ::gazer::JsonWriter json_writer;
    for (int i = 0; i < num_summary_inputs; ++i) {
      const Tensor& tensor_in = ctx->input(i);
      auto s_vec_in = tensor_in.flat<tstring>();
      for (int d = 0; d < s_vec_in.dimension(0); ++d) {
        const string& s_in = s_vec_in(d);
        Summary summary_in;
        if (!ParseProtoUnlimited(&summary_in, s_in)) {
          ctx->SetStatus(errors::InvalidArgument(
              "Could not parse one of the summary inputs"));
          return;
        }
        for (int v = 0; v < summary_in.value_size(); v++) {
          const string& tag = summary_in.value(v).tag();
          // The tag is unused by the TensorSummary op, so no need to check
          // for duplicates.
          if ((!tag.empty()) && !tags.insert(tag).second) {
            ctx->SetStatus(errors::InvalidArgument(strings::StrCat(
                "Duplicate tag ", tag, " found in summary inputs")));
            return;
          }
          metric_types.emplace_back(tag);
          metric_values.emplace_back(summary_in.value(v).simple_value());
          // *merged_summary.add_value() = summary_in.value(v);
        }
      }

      Tensor* summary_out = nullptr;
      OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output(
                              {i}, i, tensor_in.shape(), &summary_out));
    }
    json_writer.StartObject();
    json_writer.WriteVecVar("MetricType", metric_types);
    json_writer.WriteVecVar("MetricValue", metric_values);
    json_writer.EndObject();
    auto& rpc_client = ::gazer::ReportMetricsClient::GetInstance();
    rpc_client.AsyncReport(job_, task_id_, json_writer);
  }

 private:
  int num_summary_inputs;
  int task_id_;
  std::string job_;
};

REGISTER_KERNEL_BUILDER(Name("ReportAIMaster").Device(DEVICE_CPU),
                        ReportAIMasterOp);

}  // namespace gazer
}  // namespace tensorflow