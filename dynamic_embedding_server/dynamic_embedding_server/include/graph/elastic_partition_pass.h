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

#include "dynamic_embedding_server/include/utils/tensorflow_include.h"

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
  int m_num_partition;
  std::vector<Node*> m_init_op_vec;
  std::vector<Node*> m_no_op_vec;

  ElasticHookMetaNode(int num_partition)
      : m_import_op_main(nullptr),
        m_init_op_main(nullptr),
        m_tmp_value_init_op(nullptr),
        m_num_partition(num_partition),
        m_init_op_vec(num_partition, nullptr),
        m_no_op_vec(num_partition, nullptr){};
  Status Init(Graph* g, int prev_partition_ratio);
};

struct DynamicPartitionSubGraph {
  Node* dynamic_partition_node;
  Node* dynamic_stitch_node;
  Node* data_dp_node;
  Node* indices_dp_node;
  Node* new_dynamic_stitch_node;
  std::vector<Node*> gather_node_vec;
  std::vector<Node*> identity_node_vec;

  DynamicPartitionSubGraph()
      : dynamic_partition_node(nullptr),
        dynamic_stitch_node(nullptr),
        data_dp_node(nullptr),
        indices_dp_node(nullptr),
        new_dynamic_stitch_node(nullptr) {}
};

struct ConcatMeta {
    Node* concat_node;
    Node* axis_node;
    std::vector<Node*> concat_inputs;
    ConcatMeta() : concat_node(nullptr),
                   axis_node(nullptr) {}
};

class PartitionedVariable {
 public:
  PartitionedVariable()
      : g(nullptr),
        meta_node(nullptr),
        var_type(VarType::EMBEDDING_VAR),
        part_var_full_shape(0),
        ev_partition_num(0),
        cur_partition_num(0),
        variable_prefix("") {}

  PartitionedVariable(Graph* graph, ElasticHookMetaNode* m_node)
      : g(graph),
        meta_node(m_node),
        var_type(VarType::REF_VAR),
        part_var_full_shape(0),
        ev_partition_num(1) {}

  ~PartitionedVariable() {}

  void Prepare(int cur_partition_nums, int prev_partition_num,
               int delta_partition_num);

  Status Scaling(int prev_partition_ratio);

  Status ScalingUp(std::unordered_set<Node*>* nodes_to_delete,
                   int prev_partition_ratio);

  Status ScalingDown(std::unordered_set<Node*>* nodes_to_delete);

  Graph* g;
  ElasticHookMetaNode* meta_node;

  VarType var_type;
  int part_var_full_shape;
  int ev_partition_num;
  int cur_partition_num;
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
  
  void InitPartToDevice(const PartIdToNodeMap& partition_node_map,
        int& original_indices, std::vector<int>& part_to_device);

  Status ScalingUpEmbeddingVar(int original_indices,
                               std::vector<int>& part_to_device,
                               std::unordered_set<Node*>* nodes_to_delete,
                               int prev_partition_num);

  Status ScalingUpResVar(int original_indices, std::vector<int>& part_to_device,
                         std::unordered_set<Node*>* nodes_to_delete,
                         int prev_partition_num);

  Status ScalingUpVar(int original_indices, std::vector<int>& part_to_device,
                      std::unordered_set<Node*>* nodes_to_delete,
                      int prev_partition_num);

  Status ScalingDownEV(std::vector<bool>& part_to_scale_down,
                       std::unordered_set<Node*>* nodes_to_delete);

  Status ScalingDownVar(std::vector<bool>& part_to_scale_down,
                        std::unordered_set<Node*>* nodes_to_delete);

  Status ScalingDownResVar(std::vector<bool>& part_to_scale_down,
                           std::unordered_set<Node*>* nodes_to_delete);

  /********    BackwardGraph    **********/
  Status ScalingDownSparseVariableBackWardGraph(
      std::unordered_set<Node*>* nodes_to_delete,
      std::unordered_map<std::string, DynamicPartitionSubGraph>&
          prefix_name_map,
      std::vector<bool>& part_to_scale_down);

  Status ScalingDownDenseBackWardGraph(
      std::unordered_set<Node*>* nodes_to_delete,
      std::vector<bool>& part_to_scale_down);

  Status ScalingUpSparseVariableBackWardGraph(
      int original_indices,
      std::unordered_map<std::string, DynamicPartitionSubGraph>&
          prefix_name_map,
      std::vector<int>& part_to_device);

  Status ScalingUpDenseBackWardGraph(
      int original_indices,
      std::unordered_map<std::string, DynamicPartitionSubGraph>&
          prefix_name_map,
      std::vector<int>& part_to_device);

  /********    DynamicPartitionGraph    **********/
  Status RewriteDynamicPartitionGraph(
      PartIdToNodeMap& ev_node_vec,
      std::unordered_map<std::string, DynamicPartitionSubGraph>&
          prefix_name_map,
      std::unordered_set<Node*>* nodes_to_delete,
      const std::vector<bool>* part_to_scale_down = nullptr);

  /********    RedistributionGraph    **********/
  Status ScalingUpEVRedistributionGraph(
      int original_indices, PartIdToNodeMap& new_ev_node_vec,
      std::unordered_set<Node*>* nodes_to_delete,
      std::vector<int>& part_to_device,
      std::vector<std::pair<Node*, Node*>>& primary_ev_filters);

  Status ScalingUpResVarRedistributionGraph(int original_indices,
                                            PartIdToNodeMap& new_ev_node_vec,
                                            std::vector<int>& part_to_device);

  Status ScalingUpVarRedistributionGraph(int original_indices,
                                         PartIdToNodeMap& new_ev_node_vec,
                                         std::vector<int>& part_to_device);

  Status ScalingDownEVRedistributionGraph(
      PartIdToNodeMap& new_ev_node_vec,
      std::vector<std::pair<Node*, Node*>>& primary_ev_filters,
      std::unordered_set<Node*>* nodes_to_delete,
      std::vector<bool>& part_to_scale_down);

  Status ScalingDownResVarRedistributionGraph(
      PartIdToNodeMap& new_ev_node_vec,
      std::unordered_set<Node*>* nodes_to_delete,
      std::vector<bool>& part_to_scale_down);

  Status ScalingDownVarRedistributionGraph(
      PartIdToNodeMap& new_ev_node_vec,
      std::unordered_set<Node*>* nodes_to_delete,
      std::vector<bool>& part_to_scale_down);

  /********    ForwardGraph    **********/

  Status ScalingUpVarForWardGraph(int original_indices,
                                  const std::string& var_name,
                                  std::vector<int>& part_to_device,
                                  int prev_partition_ratio);

  Status ScalingUpEVForWardGraph(int original_indices,
                                 std::vector<int>& part_to_device);

  Status ScalingUpResVarForWardGraph(
      int original_indices, std::vector<int>& part_to_device,
      std::unordered_set<Node*>* nodes_to_delete);

  Status ScalingDownEVForWardGraph(std::vector<bool>& part_to_scale_down,
                                   std::unordered_set<Node*>* nodes_to_delete);

  Status ScalingDownResVarForWardGraph(
      PartIdToNodeMap& node_vec, std::vector<bool>& part_to_scale_down,
      std::unordered_set<Node*>* nodes_to_delete);

  Status ScalingDownVarForWardGraph(PartIdToNodeMap& node_vec,
                                    std::vector<bool>& part_to_scale_down,
                                    std::unordered_set<Node*>* nodes_to_delete);
};

class ElasticTrainingPass : public GraphOptimizationPass {
 public:
  ElasticTrainingPass()
      : prev_partition_num_(0),
        delta_partition_num_(0),
        scaling_up_(false),
        initialized_(false) {}
  Status Run(const GraphOptimizationPassOptions& options) override;

  Status RewriteSubGraph(Graph* g);

  Status InitVarMeta(
      Graph* g, ElasticHookMetaNode* meta_node,
      std::unordered_map<std::string, PartitionedVariable>&
          primary_ev_metas_map,
      std::unordered_map<std::string, Node*>& unpartitioned_node_map);

  // Main driver of ElasticTraining optimizations.
  Status RewriteTrainingSubGraph(
      Graph* g,
      std::unordered_map<std::string, PartitionedVariable>&
          primary_ev_metas_map,
      ElasticHookMetaNode* meta_node);

  Status RewriteSavingSubGraph(
      Graph* g,
      std::unordered_map<std::string, PartitionedVariable>&
          primary_ev_metas_map,
      std::unordered_map<std::string, Node*>& unpartitioned_node_map,
      ElasticHookMetaNode& meta_node);

 private:
  static int cur_partition_num_;
  int prev_partition_num_;
  int delta_partition_num_;
  bool scaling_up_;
  bool initialized_;
};

}  // namespace tensorflow

#endif  // DYNMAMIC_EMBEDDING_SERVER_INCLUDE_GRAPH_ELASTIC_PARTITION_PASS_H_
