#include <algorithm>
#include <unordered_map>

#include <tensorflow/core/common_runtime/optimization_registry.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/register_types.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/tensor.pb.h>
#include <tensorflow/core/graph/node_builder.h>
#include <tensorflow/core/util/device_name_utils.h>
#include "tensorflow/core/public/version.h"

namespace tensorflow {

#if (TF_MAJOR_VERSION * 1000L + TF_MINOR_VERSION) >= 1015L
constexpr char kEvInitOp[] = "InitializeKvVariableV2Op";
#else
constexpr char kEvInitOp[] = "InitializeKvVariableOp";
#endif

constexpr char kEvHandleOp[] = "KvVarHandleOp";

namespace gazer {
namespace {
constexpr char kItemSeparator[] = "@";
constexpr char kKVSeparator[] = ":";
constexpr char kGlobalResourceName[] = "GLOBAL_RESOURCE_NAME";
void VLogGraphDebugString(Graph* g) {
  GraphDef graph_def;
  g->ToGraphDef(&graph_def);
  VLOG(1) << "Grpah: " << graph_def.DebugString();
}

} // anonymous namespace

#define SINGLE_ARG(...) __VA_ARGS__
#define CASE(TYPE, STMTS)             \
  case DataTypeToEnum<TYPE>::value: { \
    typedef TYPE T;                   \
    STMTS;                            \
    break;                            \
  }
#define CASES_WITH_DEFAULT(TYPE_ENUM, STMTS, INVALID, DEFAULT) \
  switch (TYPE_ENUM) {                                         \
    CASE(float, SINGLE_ARG(STMTS))                             \
    CASE(double, SINGLE_ARG(STMTS))                            \
    CASE(int32, SINGLE_ARG(STMTS))                             \
    CASE(uint8, SINGLE_ARG(STMTS))                             \
    CASE(uint16, SINGLE_ARG(STMTS))                            \
    CASE(uint32, SINGLE_ARG(STMTS))                            \
    CASE(uint64, SINGLE_ARG(STMTS))                            \
    CASE(int16, SINGLE_ARG(STMTS))                             \
    CASE(int8, SINGLE_ARG(STMTS))                              \
    CASE(string, SINGLE_ARG(STMTS))                            \
    CASE(complex64, SINGLE_ARG(STMTS))                         \
    CASE(complex128, SINGLE_ARG(STMTS))                        \
    CASE(int64, SINGLE_ARG(STMTS))                             \
    CASE(bool, SINGLE_ARG(STMTS))                              \
    CASE(qint32, SINGLE_ARG(STMTS))                            \
    CASE(quint8, SINGLE_ARG(STMTS))                            \
    CASE(qint8, SINGLE_ARG(STMTS))                             \
    CASE(quint16, SINGLE_ARG(STMTS))                           \
    CASE(qint16, SINGLE_ARG(STMTS))                            \
    CASE(bfloat16, SINGLE_ARG(STMTS))                          \
    CASE(Eigen::half, SINGLE_ARG(STMTS))                       \
    CASE(ResourceHandle, SINGLE_ARG(STMTS))                    \
    CASE(Variant, SINGLE_ARG(STMTS))                           \
    case DT_INVALID:                                           \
      INVALID;                                                 \
      break;                                                   \
    default:                                                   \
      DEFAULT;                                                 \
      break;                                                   \
  }

#define CASES(TYPE_ENUM, STMTS)                                      \
  CASES_WITH_DEFAULT(TYPE_ENUM, STMTS, LOG(FATAL) << "Type not set"; \
                     , LOG(FATAL) << "Unexpected type: " << TYPE_ENUM;)

class ResourceCollectionPass : public GraphOptimizationPass {
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
              "ResourceCollectionPass should be optimized after partition graph,"
              " but partition graph is unavailable.");
        }
        VLogGraphDebugString(graph);

        for (Node* node : graph->op_nodes()) {
          if (node->type_string() == "VariableV2"
              || node->type_string() == "VarHandleOp") {
            if (resource_names_and_size_.end() == resource_names_and_size_.find(
                node->name())) {
              AttrSlice n_attrs = node->attrs();
              DataType dtype;
              TF_RETURN_IF_ERROR(GetNodeAttr(n_attrs, "dtype", &dtype));
              TensorShapeProto shape;
              TF_RETURN_IF_ERROR(GetNodeAttr(n_attrs, "shape", &shape));
              int64 size = 0;
              CASES(dtype, size = TensorShape(shape).num_elements() * sizeof(T));
              resource_names_and_size_.insert(std::pair<std::string, int64>(node->name(), size));
            }
          } else if (node->type_string() == "KvVarHandleOp") {
	          std::string node_name = node->name();
            for (auto* o_node: node->out_nodes()) {
              if (o_node->type_string() == kEvInitOp) {
#if (TF_MAJOR_VERSION * 1000L + TF_MINOR_VERSION) >= 1015L
                const Node* input_ev_0;
                TF_RETURN_IF_ERROR(o_node->input_node(0, &input_ev_0));
                const Node* input_ev_1;
                TF_RETURN_IF_ERROR(o_node->input_node(1, &input_ev_1));
                if (input_ev_0->name() != input_ev_1->name()) {
		  continue;
		}
#endif
                if (resource_names_and_size_.end() == resource_names_and_size_.find(node_name)) {
                  resource_names_and_size_.insert(std::pair<std::string, int64>(node_name, -1));
                }
              }
            }
          }
        }
      }
      std::string resources;
      for (const auto& r : resource_names_and_size_) {
        strings::StrAppend(&resources, r.first);
        strings::StrAppend(&resources, kKVSeparator);
        strings::StrAppend(&resources, std::to_string(r.second));
        strings::StrAppend(&resources, kItemSeparator);
      }
      setenv(kGlobalResourceName, resources.c_str(), 1);
    }
    return Status::OK();
  }
 private:
  std::unordered_map<std::string, int64> resource_names_and_size_;
};

REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_PARTITIONING, 0,
                      ResourceCollectionPass);

}  // namespace gazer
}  // namespace tensorflow
