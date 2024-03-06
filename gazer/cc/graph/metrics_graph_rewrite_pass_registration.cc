#include <queue>

#include <tensorflow/core/common_runtime/optimization_registry.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/register_types.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/tensor.pb.h>
#include <tensorflow/core/graph/node_builder.h>
#include <tensorflow/core/util/device_name_utils.h>

namespace tensorflow {
namespace gazer {
namespace {
void VLogGraphDebugString(Graph* g) {
  GraphDef graph_def;
  g->ToGraphDef(&graph_def);
  VLOG(1) << "Grpah: " << graph_def.DebugString();
}

} // anonymous namespace

class MetricsGraphRewritePass : public GraphOptimizationPass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override {
    if (options.partition_graphs) {
      for (auto& pg : *options.partition_graphs) {
        Graph* graph = pg.second.get();
        if (graph == nullptr) {
          return errors::Internal(
              "MetricsGraphRewritePass should be optimized after partition graph,"
              " but partition graph is unavailable.");
        }

        VLogGraphDebugString(graph);
        Node* time_stamp_begin_node = nullptr;
        Node* time_stamp_end_node = nullptr;
        for (Node* node : graph->nodes()) {
          if (node->type_string() == "TimeStamp") {
            if (str_util::StrContains(node->name(), "begin")) {
              time_stamp_begin_node = node;
            } else if (str_util::StrContains(node->name(), "end")) {
              time_stamp_end_node = node;
            }
          }
        }
        if (time_stamp_begin_node && time_stamp_end_node) {
          Node* source = graph->source_node();
          for (const Edge* e : source->out_edges()) {
            if (e == nullptr) {
              // for graph is null
            } else if (e->dst() == time_stamp_begin_node) {
              // to avoid generate ring
            } else {
              graph->AddControlEdge(time_stamp_begin_node, e->dst(), true);
              graph->RemoveControlEdge(e);
            }
          }
          graph->AddControlEdge(source, time_stamp_begin_node, true);

          std::set<Node*> time_stamp_end_children = GetChildren(time_stamp_end_node);

          Node* sink = graph->sink_node();
          for (const Edge* e : sink->in_edges()) {
            if (e == nullptr) {
              // for graph is null
            } else if (time_stamp_end_children.find(e->src()) != time_stamp_end_children.end()) {
              // to avoid generate ring
            } else {
              graph->AddControlEdge(e->src(), time_stamp_end_node, true);
              graph->RemoveControlEdge(e);
            }
          }
          graph->AddControlEdge(time_stamp_end_node, sink, true);

          VLogGraphDebugString(graph);
        }
      }
    }
    return Status::OK();
  }
 private:
  std::set<Node*> GetChildren(Node* n) {
    std::set<Node*> children;
    std::queue<Node*> q({n});
    while (!q.empty()) {
      Node* n = q.front();
      children.insert(n);
      for (const Edge* e : n->out_edges()) {
        q.push(e->dst());
      }
      q.pop();
    }
    return children;
  }

};

REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_PARTITIONING, 0,
                      MetricsGraphRewritePass);

}  // namespace gazer
}  // namespace tensorflow
