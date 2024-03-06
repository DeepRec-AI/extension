#include <tensorflow/core/common_runtime/optimization_registry.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/register_types.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/tensor.pb.h>
#include <tensorflow/core/graph/node_builder.h>
#include <tensorflow/core/util/device_name_utils.h>

#include <tensorflow/core/util/events_writer.h>

namespace tensorflow {
namespace gazer {
namespace {
void VLogGraphDebugString(Graph* g) {
  GraphDef graph_def;
  g->ToGraphDef(&graph_def);
  VLOG(1) << "Grpah: " << graph_def.DebugString();
}

} // anonymous namespace

class MetricsPass : public GraphOptimizationPass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override {
    if (options.graph) {
      Graph* graph = options.graph->get();
      if (graph) {
        VLogGraphDebugString(graph);
      }
    }
    if (options.partition_graphs) {
      for (auto& pg : *options.partition_graphs) {
        Graph* graph = pg.second.get();
        if (graph == nullptr) {
          return errors::Internal(
              "MetricsPass should be optimized after partition graph,"
              " but partition graph is unavailable.");
        }
        VLogGraphDebugString(graph);

        Node* merge_summary_node = nullptr;
        for (Node* node : graph->op_nodes()) {
          if (node->type_string() == "MergeSummary") {
            merge_summary_node = node;
            break;
          }
        }
        if (merge_summary_node) {
          Node* cpu_summary_node = nullptr;
          Node* mem_summary_node = nullptr;
          TF_RETURN_IF_ERROR(CreateResUtilSummaryToGraph("gazer/cpu",
                                                         "gazer/mem",
                                                         graph,
                                                         &cpu_summary_node,
                                                         &mem_summary_node));

          std::vector<Node*> summary_nodes;
          summary_nodes.emplace_back(cpu_summary_node);
          summary_nodes.emplace_back(mem_summary_node);

          TF_RETURN_IF_ERROR(ExtendMergeSummaryNodeToGraph(summary_nodes,
                                                           merge_summary_node,
                                                           graph));
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
                                     const std::string& mem_tag,
                                     Graph* g,
                                     Node** cpu_node,
                                     Node** mem_node) {
    Node* metrics_node = nullptr;
    TF_RETURN_IF_ERROR(NodeBuilder("gazer/metrics", "ResourceUtilization")
      .Finalize(g, &metrics_node));
    VLOG(1) << "create resource_utilization_node: " << metrics_node->DebugString();

    Node* cpu_const_node = nullptr;
    TF_RETURN_IF_ERROR(CreateScalarStringConstToGraph("gazer/const_1",
      cpu_tag, g, &cpu_const_node));

    Node* mem_const_node = nullptr;
    TF_RETURN_IF_ERROR(CreateScalarStringConstToGraph("gazer/const_1",
      mem_tag, g, &mem_const_node));

    TF_RETURN_IF_ERROR(NodeBuilder("gazer/summary", "ScalarSummary")
      .Input(cpu_const_node, 0)
      .Input(metrics_node, 0)
      .Attr("T", DT_FLOAT)
      .Finalize(g, cpu_node));

    TF_RETURN_IF_ERROR(NodeBuilder("gazer/summary", "ScalarSummary")
      .Input(mem_const_node, 0)
      .Input(metrics_node, 1)
      .Attr("T", DT_FLOAT)
      .Finalize(g, mem_node));

    VLOG(1) << "create cpu_node: " << (*cpu_node)->DebugString();
    VLOG(1) << "create mem_node: " << (*mem_node)->DebugString();
    return Status::OK();
  }

  Status CreateScalarStringConstToGraph(const std::string& node_name,
                                       const std::string& string_content,
                                       Graph* g,
                                       Node** const_node) {
    Tensor const_tensor(DT_STRING, TensorShape({}));
    const_tensor.scalar<std::string>()() = string_content;
    Node* node;
    TF_RETURN_IF_ERROR(NodeBuilder(node_name, "Const")
                         .Attr("dtype", DT_STRING)
                         .Attr("value", const_tensor)
                         .Finalize(g, &node));
    VLOG(1) << "create const_node: " << node->DebugString();
    *const_node = node;
    return Status::OK();
  }

  Status ExtendMergeSummaryNodeToGraph(std::vector<Node*> summary_nodes,
                                       Node* merge_summary_node,
                                       Graph* g) {
    int64 n = 0;
    TF_RETURN_IF_ERROR(GetNodeAttr(merge_summary_node->attrs(), "N", &n));
    NodeBuilder merge_summary_builder = NodeBuilder(merge_summary_node->name(),
      merge_summary_node->type_string()).Attr<int64>("N", n + summary_nodes.size());

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
    TF_RETURN_IF_ERROR(merge_summary_builder.Finalize(g, &new_merge_summary_node));
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
    return Status::OK();
  }
};

REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_PARTITIONING, 0,
                      MetricsPass);

}  // namespace gazer
}  // namespace tensorflow
