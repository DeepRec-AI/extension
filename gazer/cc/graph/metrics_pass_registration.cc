#include <grpc/grpc.h>
#include <tensorflow/core/common_runtime/optimization_registry.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/register_types.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/tensor.pb.h>
#include <tensorflow/core/graph/algorithm.h>
#include <tensorflow/core/graph/node_builder.h>
#include <tensorflow/core/util/device_name_utils.h>
#include <tensorflow/core/util/env_var.h>

#include "gazer/cc/client/grpc_client.h"

namespace tensorflow {
namespace gazer {
namespace {
void VLogGraphDebugString(Graph* g) {
  GraphDef graph_def;
  g->ToGraphDef(&graph_def);
  VLOG(1) << "Grpah: " << graph_def.DebugString();
}

}  // anonymous namespace

class MetricsPass : public GraphOptimizationPass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override {
    if (options.graph) {
      Graph* graph = options.graph->get();
      if (graph) {
        VLogGraphDebugString(graph);
        std::vector<Device*> all_devices;
        for (auto& d : options.device_set->devices()) {
          VLOG(1) << "device: " << d->DebugString();
          VLOG(1) << "device name: " << d->name();
          // filter CPU device, remove GPU, XLA_CPU ... device
          if (d->device_type() == "CPU") {
            all_devices.emplace_back(d);
          }
        }
        CHECK(all_devices.size() > 0);

        Node* merge_summary_node = nullptr;
        for (Node* node : graph->op_nodes()) {
          if (node->type_string() == "MergeSummary") {
            merge_summary_node = node;
            break;
          }
        }
        if (merge_summary_node) {
          std::vector<Node*> summary_nodes;
          int task_id = 0;
          std::string job;
          for (auto& d : all_devices) {
            int assigned_device_index = graph->InternDeviceName(d->name());
            std::string node_name_suffix =
                str_util::StringReplace(d->name(), ":", "_", true);
            Node* cpu_summary_node = nullptr;
            Node* mem_summary_node = nullptr;
            TF_RETURN_IF_ERROR(CreateResUtilSummaryToGraph(
                "gazer/cpu" + node_name_suffix, "gazer/mem" + node_name_suffix,
                graph, assigned_device_index, node_name_suffix,
                &cpu_summary_node, &mem_summary_node));

            Node* res_summary_node = nullptr;
            TF_RETURN_IF_ERROR(CreateResSummaryToGraph(
                "gazer/res_mem" + node_name_suffix, graph,
                assigned_device_index, node_name_suffix, &res_summary_node));
            Node* graph_summary_node = nullptr;
            TF_RETURN_IF_ERROR(CreateGraphSummaryToGraph(
                "gazer/graph_duration" + node_name_suffix, graph,
                assigned_device_index, node_name_suffix, &graph_summary_node));

            summary_nodes.emplace_back(cpu_summary_node);
            summary_nodes.emplace_back(mem_summary_node);
            summary_nodes.emplace_back(res_summary_node);
            summary_nodes.emplace_back(graph_summary_node);
            job = d->parsed_name().job;
            task_id = d->parsed_name().task;
          }

          // Read AIMASTER ENV
          std::string ai_master_addr;
          ReadStringFromEnvVar("AIMASTER_ADDR", "", &ai_master_addr);
          if (ai_master_addr != "") {
            auto& client = ::gazer::ReportMetricsClient::GetInstance();
            LOG(INFO) << "INITIALIZED gazer... AIMASTER_ADDR: "
                      << ai_master_addr;
            TF_RETURN_IF_ERROR(client.Initialize(ai_master_addr));
            TF_RETURN_IF_ERROR(client.ConnectToAM(job, task_id));
            TF_RETURN_IF_ERROR(ExtendReportAIMasterNodeToGraph(
                summary_nodes, job, task_id, merge_summary_node, graph));
          } else {
            TF_RETURN_IF_ERROR(ExtendMergeSummaryNodeToGraph(
                summary_nodes, merge_summary_node, graph));
          }
          VLogGraphDebugString(graph);
        } else {
          // MergeSummary not found, do nothing
          VLOG(1) << "MergeSummary not found, do nothing";
        }
      }
    }
    return Status::OK();
  }

 private:
  Status CreateResUtilSummaryToGraph(const std::string& cpu_tag,
                                     const std::string& mem_tag, Graph* g,
                                     int assigned_device_index,
                                     const std::string& name_suffix,
                                     Node** cpu_node, Node** mem_node) {
    Node* metrics_node = nullptr;
    TF_RETURN_IF_ERROR(NodeBuilder("gazer/resource_utilization" + name_suffix,
                                   "ResourceUtilization")
                           .Finalize(g, &metrics_node));
    metrics_node->set_assigned_device_name_index(assigned_device_index);
    VLOG(1) << "create resource_utilization_node: "
            << metrics_node->DebugString();

    Node* cpu_const_node = nullptr;
    TF_RETURN_IF_ERROR(CreateScalarStringConstToGraph(
        "gazer/const" + name_suffix + "/0", cpu_tag, g, assigned_device_index,
        &cpu_const_node));

    Node* mem_const_node = nullptr;
    TF_RETURN_IF_ERROR(CreateScalarStringConstToGraph(
        "gazer/const" + name_suffix + "/1", mem_tag, g, assigned_device_index,
        &mem_const_node));

    TF_RETURN_IF_ERROR(
        NodeBuilder("gazer/summary" + name_suffix + "/0", "ScalarSummary")
            .Input(cpu_const_node, 0)
            .Input(metrics_node, 0)
            .Attr("T", DT_FLOAT)
            .Finalize(g, cpu_node));
    (*cpu_node)->set_assigned_device_name_index(assigned_device_index);

    TF_RETURN_IF_ERROR(
        NodeBuilder("gazer/summary" + name_suffix + "/1", "ScalarSummary")
            .Input(mem_const_node, 0)
            .Input(metrics_node, 1)
            .Attr("T", DT_FLOAT)
            .Finalize(g, mem_node));
    (*mem_node)->set_assigned_device_name_index(assigned_device_index);

    VLOG(1) << "create cpu_node: " << (*cpu_node)->DebugString();
    VLOG(1) << "create mem_node: " << (*mem_node)->DebugString();
    return Status::OK();
  }

  Status CreateGraphSummaryToGraph(const std::string& tag, Graph* g,
                                   int assigned_device_index,
                                   const std::string& name_suffix,
                                   Node** node) {
    Node* time_stamp_begin = nullptr;
    TF_RETURN_IF_ERROR(
        NodeBuilder("gazer/graph_stat/begin" + name_suffix, "TimeStamp")
            .Finalize(g, &time_stamp_begin));
    time_stamp_begin->set_assigned_device_name_index(assigned_device_index);

    Node* time_stamp_end = nullptr;
    TF_RETURN_IF_ERROR(
        NodeBuilder("gazer/graph_stat/end" + name_suffix, "TimeStamp")
            .Finalize(g, &time_stamp_end));
    time_stamp_end->set_assigned_device_name_index(assigned_device_index);

    Node* const_node = nullptr;
    TF_RETURN_IF_ERROR(
        CreateScalarStringConstToGraph("gazer/graph/const" + name_suffix, tag,
                                       g, assigned_device_index, &const_node));

    Node* duartion_node = nullptr;
    TF_RETURN_IF_ERROR(
        NodeBuilder("gazer/graph_stat/duration" + name_suffix, "Sub")
            .Input(time_stamp_end, 0)
            .Input(time_stamp_begin, 0)
            .Finalize(g, &duartion_node));
    duartion_node->set_assigned_device_name_index(assigned_device_index);

    TF_RETURN_IF_ERROR(
        NodeBuilder("gazer/graph_summary" + name_suffix, "ScalarSummary")
            .Input(const_node, 0)
            .Input(duartion_node, 0)
            .Finalize(g, node));
    (*node)->set_assigned_device_name_index(assigned_device_index);
    return Status::OK();
  }

  Status CreateResSummaryToGraph(const std::string& tag, Graph* g,
                                 int assigned_device_index,
                                 const std::string& name_suffix, Node** node) {
    Node* const_node = nullptr;
    TF_RETURN_IF_ERROR(
        CreateScalarStringConstToGraph("gazer/const" + name_suffix, tag, g,
                                       assigned_device_index, &const_node));

    Node* metrics_node = nullptr;
    TF_RETURN_IF_ERROR(
        NodeBuilder("gazer/resource_stat" + name_suffix, "ResourceStat")
            .Finalize(g, &metrics_node));
    metrics_node->set_assigned_device_name_index(assigned_device_index);
    VLOG(1) << "create resource_utilization_node: "
            << metrics_node->DebugString();

    TF_RETURN_IF_ERROR(
        NodeBuilder("gazer/res_summary" + name_suffix, "ScalarSummary")
            .Input(const_node, 0)
            .Input(metrics_node, 0)
            .Attr("T", DT_FLOAT)
            .Finalize(g, node));
    (*node)->set_assigned_device_name_index(assigned_device_index);
    VLOG(1) << "create summary_node: " << (*node)->DebugString();
    return Status::OK();
  }

  Status CreateScalarStringConstToGraph(const std::string& node_name,
                                        const std::string& string_content,
                                        Graph* g, int assigned_device_index,
                                        Node** const_node) {
    Tensor const_tensor(DT_STRING, TensorShape({}));
    const_tensor.scalar<std::string>()() = string_content;
    Node* node;
    TF_RETURN_IF_ERROR(NodeBuilder(node_name, "Const")
                           .Attr("dtype", DT_STRING)
                           .Attr("value", const_tensor)
                           .Finalize(g, &node));
    node->set_assigned_device_name_index(assigned_device_index);
    VLOG(1) << "create const_node: " << node->DebugString();
    *const_node = node;
    return Status::OK();
  }

  Status ExtendReportAIMasterNodeToGraph(
      const std::vector<Node*>& summary_nodes, const std::string& job,
      int task_id, Node* merge_summary_node, Graph* g) {
    NodeBuilder report_aimaster_builder =
        NodeBuilder("report_aimaster", "ReportAIMaster")
            .Attr<int64>("N", summary_nodes.size())
            .Attr<string>("job", job.c_str())
            .Attr<int64>("task_id", task_id);
    std::vector<NodeBuilder::NodeOut> rpc_nodes_srcs;
    for (const auto& summary_node : summary_nodes) {
      rpc_nodes_srcs.emplace_back(NodeBuilder::NodeOut(summary_node));
    }
    report_aimaster_builder.Input(rpc_nodes_srcs);
    Node* report_aimaster_node = nullptr;
    TF_RETURN_IF_ERROR(
        report_aimaster_builder.Finalize(g, &report_aimaster_node));

    int n = 0;
    TF_RETURN_IF_ERROR(GetNodeAttr(merge_summary_node->attrs(), "N", &n));
    std::vector<NodeBuilder::NodeOut> srcs;
    srcs.reserve(n + report_aimaster_node->num_outputs());

    for (const auto& edge : merge_summary_node->in_edges()) {
      if (edge->IsControlEdge()) {
        continue;
      }
      Node* src_node = edge->src();
      srcs.emplace_back(src_node);
    }
    for (int i = 0; i < report_aimaster_node->num_outputs(); ++i) {
      srcs.emplace_back(report_aimaster_node, i);
    }

    NodeBuilder merge_summary_builder =
        NodeBuilder(merge_summary_node->name(),
                    merge_summary_node->type_string())
            .Attr<int64>("N", srcs.size());

    merge_summary_builder.Input(srcs);
    Node* new_merge_summary_node = nullptr;
    TF_RETURN_IF_ERROR(
        merge_summary_builder.Finalize(g, &new_merge_summary_node));

    new_merge_summary_node->set_assigned_device_name_index(
        merge_summary_node->assigned_device_name_index());

    report_aimaster_node->set_assigned_device_name_index(
        merge_summary_node->assigned_device_name_index());

    for (const auto& edge : merge_summary_node->out_edges()) {
      if (edge->dst_input() == Graph::kControlSlot) {
        g->AddControlEdge(new_merge_summary_node, edge->dst());
      } else {
        g->AddEdge(new_merge_summary_node, edge->src_output(), edge->dst(),
                   edge->dst_input());
      }
    }

    g->RemoveNode(merge_summary_node);
    FixupSourceAndSinkEdges(g);
    return Status::OK();
  }

  Status ExtendMergeSummaryNodeToGraph(const std::vector<Node*>& summary_nodes,
                                       Node* merge_summary_node, Graph* g) {
    int64 n = 0;
    TF_RETURN_IF_ERROR(GetNodeAttr(merge_summary_node->attrs(), "N", &n));
    NodeBuilder merge_summary_builder =
        NodeBuilder(merge_summary_node->name(),
                    merge_summary_node->type_string())
            .Attr<int64>("N", n + summary_nodes.size());

    std::vector<NodeBuilder::NodeOut> srcs;
    srcs.reserve(n + summary_nodes.size());
    for (const auto& edge : merge_summary_node->in_edges()) {
      if (edge->IsControlEdge()) {
        continue;
      }
      Node* src_node = edge->src();
      srcs.emplace_back(src_node);
    }
    for (const auto& summary_node : summary_nodes) {
      srcs.emplace_back(NodeBuilder::NodeOut(summary_node));
    }
    merge_summary_builder.Input(srcs);

    Node* new_merge_summary_node = nullptr;
    TF_RETURN_IF_ERROR(
        merge_summary_builder.Finalize(g, &new_merge_summary_node));
    new_merge_summary_node->set_assigned_device_name_index(
        merge_summary_node->assigned_device_name_index());

    for (const auto& edge : merge_summary_node->out_edges()) {
      if (edge->dst_input() == Graph::kControlSlot) {
        g->AddControlEdge(new_merge_summary_node, edge->dst());
      } else {
        g->AddEdge(new_merge_summary_node, edge->src_output(), edge->dst(),
                   edge->dst_input());
      }
    }
    g->RemoveNode(merge_summary_node);
    FixupSourceAndSinkEdges(g);
    return Status::OK();
  }
};

REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, 0,
                      MetricsPass);

}  // namespace gazer
}  // namespace tensorflow
