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

#ifndef DYNMAMIC_EMBEDDING_SERVER_INCLUDE_GRAPH_ELASTIC_PARTITION_PASS_H_
#define DYNMAMIC_EMBEDDING_SERVER_INCLUDE_GRAPH_ELASTIC_PARTITION_PASS_H_

#include <sstream>

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"

namespace tensorflow {

typedef std::unordered_map<int, Node*> PartIdToNodeMap;
typedef std::pair<int, Node*> DeviceIdToNodePair;

enum VarType {
  EMBEDDING_VAR = 1,
  RESOURCE_VAR = 2,
  REF_VAR = 3,
  DENSE_RESOUCE_VAR = 4,
  DENSE_REF_VAR = 5,
};

struct ElasticHookMetaNode {
  Node* m_import_op_main;
  Node* m_init_op_main;
  Node* m_tmp_value_init_op;
  std::vector<Node*> m_init_op_vec;

  ElasticHookMetaNode(int num_partition)
      : m_import_op_main(nullptr),
        m_init_op_main(nullptr),
        m_tmp_value_init_op(nullptr),
        m_init_op_vec(num_partition, nullptr){};
};

class PartitionedVariable {
 public:
  PartitionedVariable()
      : g(nullptr),
        meta_node(nullptr),
        var_type(VarType::EMBEDDING_VAR),
        part_var_full_shape(0),
        ev_partition_num(0),
        cur_partition_nums_(0),
        variable_prefix("") {}

  PartitionedVariable(Graph* graph, ElasticHookMetaNode* m_node)
      : g(graph),
        meta_node(m_node),
        var_type(VarType::REF_VAR),
        part_var_full_shape(0),
        ev_partition_num(1) {}

  ~PartitionedVariable() {}

  void Prepare(int prev_partition_ratio);

  Status Scaling(int cur_partition_nums, int prev_partition_ratio);

  Status ScalingUp(std::unordered_set<Node*>& nodes_to_delete,
                   int prev_partition_ratio);

  Status ScalingDown(std::unordered_set<Node*>& nodes_to_delete);

  Graph* g;
  ElasticHookMetaNode* meta_node;

  VarType var_type;
  int part_var_full_shape;
  int ev_partition_num;
  int cur_partition_nums_;
  std::string variable_prefix;
  std::unordered_set<std::string> opt_ev_names;
  std::vector<std::string> sorted_opt_ev_names;
  std::unordered_map<Node*, int> node_device_map;
  std::unordered_map<std::string, PartIdToNodeMap> node_map;

 private:
  std::string DebugString() {
    std::stringstream debug_string;
    debug_string << "processing: " << variable_prefix << " var_type "
                 << var_type << " opt_name size: " << sorted_opt_ev_names.size()
                 << " ev_num: " << ev_partition_num << "\n";
    for (auto& it : node_device_map) {
      debug_string << it.first->name() << " device_id: " << it.second << "\n";
    }
    return debug_string.str();
  }

  Status ScalingDownSparseVariableBackWardGraph(
      std::unordered_set<Node*>& nodes_to_delete, Node* data_dp_node,
      Node* indices_dp_node, Node* p_dynamic_stitch_node,
      std::vector<bool>& part_to_scale_down);

  Status ScalingDownDenseBackWardGraph(
      std::unordered_set<Node*>& nodes_to_delete,
      std::vector<bool>& part_to_scale_down);

  Status ScalingUpSparseVariableBackWardGraph(int original_indices,
                                              Node* data_dp_node,
                                              Node* indices_dp_node,
                                              Node* p_dynamic_stitch_node,
                                              std::vector<Node*>& no_op_vec,
                                              std::vector<int>& part_to_device);

  Status ScalingUpDenseBackWardGraph(int original_indices, Node* data_dp_node,
                                     Node* indices_dp_node,
                                     std::vector<Node*>& no_op_vec,
                                     std::vector<int>& part_to_device);

  Status ScalingDownBackWardGraph(std::unordered_set<Node*>& nodes_to_delete,
                                  Node* data_dp_node, Node* indices_dp_node,
                                  Node* p_dynamic_stitch_node,
                                  std::vector<bool>& part_to_scale_down);

  Status ScalingUpBackWardGraph(int original_indices, Node* data_dp_node,
                                Node* indices_dp_node,
                                Node* p_dynamic_stitch_node,
                                std::vector<Node*>& no_op_vec,
                                std::vector<int>& part_to_device);

  Status RewriteElasticPartitionGraph(
      PartIdToNodeMap& ev_node_vec, Node** data_dp_node, Node** indices_dp_node,
      Node** p_dynamic_stitch_node, std::unordered_set<Node*>& nodes_to_delete);

  Status ScalingUpRedistributionGraph(int original_indices,
                                      PartIdToNodeMap& new_ev_node_vec,
                                      Node* import_op_main,
                                      std::vector<int>& part_to_device,
                                      std::vector<Node*>& primary_ev_filters);

  Status ScalingDownRedistributionGraph(
      VarType& var_type, PartIdToNodeMap& new_ev_node_vec,
      std::unordered_set<Node*>& nodes_to_delete,
      std::vector<bool>& part_to_scale_down);

  Status ScalingUpEVRedistributionGraph(int original_indices,
                                        PartIdToNodeMap& new_ev_node_vec,
                                        Node* import_op_main,
                                        std::vector<int>& part_to_device,
                                        std::vector<Node*>& primary_ev_filters);

  Status ScalingUpResVarRedistributionGraph(
      int original_indices, PartIdToNodeMap& new_ev_node_vec,
      Node* import_op_main, std::vector<int>& part_to_device,
      std::vector<Node*>& primary_ev_filters);

  Status ScalingUpVarRedistributionGraph(
      int original_indices, PartIdToNodeMap& new_ev_node_vec,
      Node* import_op_main, std::vector<int>& part_to_device,
      std::vector<Node*>& primary_ev_filters);

  Status ScalingDownEVRedistributionGraph(
      PartIdToNodeMap& new_ev_node_vec,
      std::unordered_set<Node*>& nodes_to_delete,
      std::vector<bool>& part_to_scale_down);

  Status ScalingDownResVarRedistributionGraph(
      PartIdToNodeMap& new_ev_node_vec,
      std::unordered_set<Node*>& nodes_to_delete,
      std::vector<bool>& part_to_scale_down);

  Status ScalingDownVarRedistributionGraph(
      PartIdToNodeMap& new_ev_node_vec,
      std::unordered_set<Node*>& nodes_to_delete,
      std::vector<bool>& part_to_scale_down);

  Status ScalingUpForWardGraph(int original_indices,
                               std::vector<int>& part_to_device,
                               std::unordered_set<Node*>& nodes_to_delete,
                               int prev_partition_ratio);

  Status ScalingDownForWardGraph(std::vector<bool>& part_to_scale_down,
                                 std::unordered_set<Node*>& nodes_to_delete);

  Status ScalingUpVarForWardGraph(int original_indices,
                                  const std::string& var_name,
                                  std::vector<int>& part_to_device,
                                  int prev_partition_ratio);

  Status ScalingUpEVForWardGraph(int original_indices,
                                 std::vector<int>& part_to_device,
                                 std::unordered_set<Node*>& nodes_to_delete);

  Status ScalingUpResVarForWardGraph(
      int original_indices, std::vector<int>& part_to_device,
      std::unordered_set<Node*>& nodes_to_delete);

  Status ScalingDownEVForWardGraph(std::vector<bool>& part_to_scale_down,
                                   std::unordered_set<Node*>& nodes_to_delete);

  Status ScalingDownResVarForWardGraph(
      PartIdToNodeMap& node_vec, std::vector<bool>& part_to_scale_down,
      std::unordered_set<Node*>& nodes_to_delete);

  Status ScalingDownVarForWardGraph(PartIdToNodeMap& node_vec,
                                    std::vector<bool>& part_to_scale_down,
                                    std::unordered_set<Node*>& nodes_to_delete);
};

class ElasticTrainingPass : public GraphOptimizationPass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override;

  Status RewriteSubGraph(Graph* g);

  Status InitHookMetaNode(Graph* g, ElasticHookMetaNode& meta_node);

  Status InitVarMeta(
      Graph* g, ElasticHookMetaNode* meta_node,
      std::unordered_map<std::string, PartitionedVariable>&
          primary_ev_metas_map,
      std::unordered_map<std::string, Node*>& unpartitioned_node_map);

  Status RewriteTrainingSubGraph(
      Graph* g,
      std::unordered_map<std::string, PartitionedVariable>&
          primary_ev_metas_map,
      ElasticHookMetaNode& meta_node);

  Status RewriteSavingSubGraph(
      Graph* g,
      std::unordered_map<std::string, PartitionedVariable>&
          primary_ev_metas_map,
      std::unordered_map<std::string, Node*>& unpartitioned_node_map,
      ElasticHookMetaNode& meta_node);

 private:
  static int cur_partition_nums_;
  int prev_partition_ratio_;
  bool scaling_up_{false};
  bool initialized_{false};
};

}  // namespace tensorflow

#endif  // DYNMAMIC_EMBEDDING_SERVER_INCLUDE_GRAPH_ELASTIC_PARTITION_PASS_H_
