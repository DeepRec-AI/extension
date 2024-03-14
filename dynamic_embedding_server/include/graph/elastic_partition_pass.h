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

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

enum VarType {
  EMBEDDING_VAR = 1,
  RESOURCE_VAR = 2,
  REF_VAR = 3,
  DENSE_RESOUCE_VAR = 4,
  DENSE_REF_VAR = 5,
};

struct PartitionVarMeta {
  VarType m_var_type;
  int m_full_shape;
  int m_partition_num;
  PartitionVarMeta()
      : m_var_type(VarType::REF_VAR),
        m_full_shape(0),
        m_partition_num(1){};
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

class ElasticTrainingPass : public GraphOptimizationPass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override;

  Status RewriteSubGraph(Graph* g, bool is_test = false);

  Status InitHookMetaNode(Graph* g, ElasticHookMetaNode& meta_node);

  Status InitVarMeta(
      Graph* g,
      std::unordered_map<std::string, PartitionVarMeta>& primary_ev_metas_map,
      std::unordered_map<std::string, std::vector<std::string>>&
          primary_ev_to_opt_map,
      std::unordered_map<std::string, std::vector<Node*>>& ev_to_origin_map,
      std::unordered_map<std::string, Node*>& unpartitioned_node_map);

  Status RewriteTrainingSubGraph(
      Graph* g,
      std::unordered_map<std::string, PartitionVarMeta>& primary_ev_metas_map,
      std::unordered_map<std::string, std::vector<std::string>>&
          primary_ev_to_opt_map,
      std::unordered_map<std::string, std::vector<Node*>>& ev_to_origin_map,
      ElasticHookMetaNode& meta_node, bool is_test);

  Status RewriteSavingSubGraph(
      Graph* g,
      std::unordered_map<std::string, PartitionVarMeta>& primary_ev_metas_map,
      std::unordered_map<std::string, std::vector<std::string>>&
          primary_ev_to_opt_map,
      std::unordered_map<std::string, std::vector<Node*>>& ev_to_origin_map,
      std::unordered_map<std::string, Node*>& unpartitioned_node_map,
      ElasticHookMetaNode& meta_node);

 private:
  Status ScalingDownEmbeddingVariableBackWardGraph(
      VarType var_type, Graph* g,
      std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
      std::unordered_set<Node*>& nodes_to_delete,
      const std::string& primary_ev_name,
      const std::vector<std::string>& opt_ev_names, Node* elastic_node,
      Node* p_dynamic_stitch_node, int part_var_full_shape,
      int ev_partition_num);

  Status ScalingDownDenseBackWardGraph(
      VarType var_type, Graph* g,
      std::unordered_map<std::string, std::vector<Node*>>& ev_to_origin_map,
      std::unordered_set<Node*>& nodes_to_delete,
      const std::string& primary_ev_name, Node* elastic_node,
      int part_var_full_shape, int ev_partition_num);

  Status ScalingUpEmbeddingVariableBackWardGraph(
      VarType var_type, Graph* g,
      std::unordered_map<std::string, std::vector<Node*>>& ev_to_origin_map,
      const std::string& primary_ev_name,
      const std::vector<std::string>& opt_ev_names, Node* elastic_node,
      Node* p_dynamic_stitch_node, std::vector<Node*>& no_op_vec,
      int part_var_full_shape, int ev_partition_num);

  Status ScalingUpDenseBackWardGraph(
      VarType var_type, Graph* g,
      std::unordered_map<std::string, std::vector<Node*>>& ev_to_origin_map,
      const std::string& primary_ev_name,
      const std::vector<std::string>& opt_ev_names, Node* elastic_node,
      std::vector<Node*>& no_op_vec, int part_var_full_shape,
      int ev_partition_num);

  Status ScalingDownBackWardGraph(
      Graph* g, VarType var_type,
      std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
      std::unordered_set<Node*>& nodes_to_delete,
      const std::string& primary_ev_name,
      const std::vector<std::string>& opt_ev_names, Node* elastic_node,
      Node* p_dynamic_stitch_node, int part_var_full_shape,
      int ev_partition_num);

  Status ScalingUpBackWardGraph(
      VarType var_type, Graph* g,
      std::unordered_map<std::string, std::vector<Node*>>& ev_to_origin_map,
      const std::string& primary_ev_name,
      const std::vector<std::string>& opt_ev_names, Node* elastic_node,
      Node* p_dynamic_stitch_node, std::vector<Node*>& no_op_vec,
      int part_var_full_shape, int ev_partition_num);

  Status RewriteElasticPartitionGraph(
      VarType var_type, Graph* g, std::vector<Node*>& ev_node_vec,
      Node** elastic_node, Node** p_dynamic_stitch_node,
      std::unordered_set<Node*>& nodes_to_delete);

  Status ScalingUpRedistributionGraph(VarType var_type, Graph* g,
                                      std::vector<Node*>& new_ev_node_vec,
                                      Node* import_op_main,
                                      int ev_partition_num,
                                      std::vector<Node*>& primary_ev_filters);

  Status ScalingDownRedistributionGraph(
      VarType& var_type, Graph* g, std::vector<Node*>& new_ev_node_vec,
      std::unordered_set<Node*>& nodes_to_delete, int ev_partition_num);

  Status ScalingUpEVRedistributionGraph(VarType var_type, Graph* g,
                                        std::vector<Node*>& new_ev_node_vec,
                                        Node* import_op_main,
                                        int ev_partition_num,
                                        std::vector<Node*>& primary_ev_filters);

  Status ScalingUpResVarRedistributionGraph(
      VarType var_type, Graph* g, std::vector<Node*>& new_ev_node_vec,
      Node* import_op_main, int ev_partition_num,
      std::vector<Node*>& primary_ev_filters);

  Status ScalingUpVarRedistributionGraph(
      VarType var_type, Graph* g, std::vector<Node*>& new_ev_node_vec,
      Node* import_op_main, int ev_partition_num,
      std::vector<Node*>& primary_ev_filters);

  Status ScalingDownEVRedistributionGraph(
      Graph* g, std::vector<Node*>& new_ev_node_vec,
      std::unordered_set<Node*>& nodes_to_delete, int ev_partition_num);

  Status ScalingDownResVarRedistributionGraph(
      Graph* g, std::vector<Node*>& new_ev_node_vec,
      std::unordered_set<Node*>& nodes_to_delete, int ev_partition_num);

  Status ScalingDownVarRedistributionGraph(
      Graph* g, std::vector<Node*>& new_ev_node_vec,
      std::unordered_set<Node*>& nodes_to_delete, int ev_partition_num);

  Status ScalingUpForWardGraph(
      const VarType& var_type, Graph* g, int part_var_full_shape,
      int ev_partition_num, const std::string& primary_variable_name,
      const std::vector<std::string>& opt_ev_names,
      ElasticHookMetaNode& meta_node,
      std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
      std::unordered_set<Node*>& nodes_to_delete);

  Status ScalingDownForWardGraph(
      const VarType& var_type, Graph* g, int part_var_full_shape,
      int ev_partition_num, const std::string& primary_ev_name,
      const std::vector<std::string>& opt_ev_names,
      std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
      std::unordered_set<Node*>& nodes_to_delete);

  Status ScalingUpVarForWardGraph(
      const VarType& var_type, Graph* g, int ev_partition_num,
      const std::string& primary_variable_name,
      const std::vector<std::string>& opt_ev_names,
      ElasticHookMetaNode& meta_node,
      std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
      std::unordered_set<Node*>& nodes_to_delete);

  Status ScalingUpEVForWardGraph(
      const VarType& var_type, Graph* g, int ev_partition_num,
      const std::string& primary_variable_name,
      const std::vector<std::string>& opt_ev_names,
      ElasticHookMetaNode& meta_node,
      std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
      std::unordered_set<Node*>& nodes_to_delete);

  Status ScalingUpResVarForWardGraph(
      const VarType& var_type, Graph* g, int part_var_full_shape,
      int ev_partition_num, const std::string& primary_variable_name,
      const std::vector<std::string>& opt_ev_names,
      ElasticHookMetaNode& meta_node,
      std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
      std::unordered_set<Node*>& nodes_to_delete);

  Status ScalingDownEVForWardGraph(
      const VarType& var_type, Graph* g, int ev_partition_num,
      const std::string& primary_variable_name,
      const std::vector<std::string>& opt_ev_names,
      std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
      std::unordered_set<Node*>& nodes_to_delete);

  Status ScalingDownResVarForWardGraph(
      const VarType& var_type, Graph* g, int part_var_full_shape,
      int ev_partition_num, const std::string& primary_variable_name,
      const std::vector<std::string>& opt_ev_names,
      std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
      std::unordered_set<Node*>& nodes_to_delete);

  Status ScalingDownVarForWardGraph(
      const VarType& var_type, Graph* g, int ev_partition_num,
      const std::string& primary_variable_name,
      const std::vector<std::string>& opt_ev_names,
      std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
      std::unordered_set<Node*>& nodes_to_delete);

 private:
  static int cur_partition_nums_;
  bool scaling_up_{false};
};

}  // namespace tensorflow

#endif  // DYNMAMIC_EMBEDDING_SERVER_INCLUDE_GRAPH_ELASTIC_PARTITION_PASS_H_