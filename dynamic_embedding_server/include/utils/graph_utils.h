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

#ifndef DYNMAMIC_EMBEDDING_SERVER_INCLUDE_UTILS_GRAPH_UTILS_H_
#define DYNMAMIC_EMBEDDING_SERVER_INCLUDE_UTILS_GRAPH_UTILS_H_

#include "dynamic_embedding_server/include/utils/naming.h"

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/strings/str_util.h"

constexpr char kPart[] = "part_";
constexpr char kDynamicPartition[] = "DynamicPartition";
constexpr char kParaDynamicStitch[] = "ParallelDynamicStitch";
constexpr char kIdentityOp[] = "Identity";
constexpr char kVariableOp[] = "VariableV2";
constexpr char kSaveOp[] = "SaveV3";
constexpr char kRestoreOp[] = "RestoreV2";
constexpr char kWorkerSyncOp[] = "worker_sync";

namespace tensorflow {

typedef std::pair<int, Node*> DeviceIdToNode;

inline string NewDevice(Node* node, int i) {
  DeviceNameUtils::ParsedName full_device_name;
  DeviceNameUtils::ParseFullName(node->assigned_device_name(),
                                 &full_device_name);
  full_device_name.task = i;
  return DeviceNameUtils::ParsedNameToString(full_device_name);
}

Status FindNodeAndExec(Node* src, const string& target,
                       std::function<Status(Node* target_node)> fn) {
  for (auto* o_node : src->out_nodes()) {
    if (o_node->type_string() == target) {
      fn(o_node);
    }
  }
  return Status::OK();
}

inline string NewNodeName(const string& ori_name, int partition_id) {
  auto part_idx = ori_name.find(kPart);
  if (part_idx == -1) {
    return ori_name + "/Copy_" + std::to_string(partition_id);
  } else {
    std::string pre_str = ori_name.substr(0, part_idx - 1);
    std::string post_str = ori_name.substr(part_idx + strlen(kPart));
    auto post_idx = post_str.find("/");
    if (post_idx == string::npos) {
      return pre_str + "/" + kPart + std::to_string(partition_id);
    } else {
      return pre_str + "/" + kPart + std::to_string(partition_id) +
             post_str.substr(post_idx);
    }
  }
}

inline int GetNodePartIndex(Node* node) {
  string node_name = node->name();
  size_t pos = node_name.find("/part_");
  if (pos == std::string::npos) {
    LOG(ERROR) << "Invalid string format";
    return -1;
  }

  std::string partNumberStr = node_name.substr(pos + 6);
  size_t slash_pos = partNumberStr.find("//");
  if (slash_pos == string::npos) {
    return std::stoi(partNumberStr);
  } else {
    std::string NumberStr = node_name.substr(0, slash_pos);
    return std::stoi(NumberStr);
  }
  return -1;
}

inline Node* CopyNode(Graph* g, const Node* node,
                      const std::string& device_name, int partition_id,
                      const string& op_name = "") {
  Node* ret = g->CopyNode(node);
  if (op_name == "") {
    ret->set_name(NewNodeName(node->name(), partition_id));
  } else {
    ret->set_name(op_name);
  }
  ret->set_assigned_device_name(device_name);
  ret->ClearAttr("_class");  // remove pre-exist colocation
  return std::move(ret);
}

inline bool IsApplyNode(VarType var_type, Node* node) {
  switch (var_type) {
    case VarType::EMBEDDING_VAR:
      return node->IsKvSparseApply();
      break;
    case VarType::REF_VAR:
    case VarType::RESOURCE_VAR:
      return node->IsSparseApplyAdagradOps() || node->IsSparseApplyFtrlOps() ||
             node->IsApplySparseAdamOps();
      break;
    case VarType::DENSE_RESOUCE_VAR:
    case VarType::DENSE_REF_VAR:
      return node->IsApplyAdamOps() || node->IsApplyAdagradOps() ||
             node->IsApplyFtrlOps();
  }
  return false;
}

Status ChangeResShape(Node* res_node, TensorShape& new_shape,
                      int part_var_full_shape, int cur_partition_nums,
                      bool last_part) {
  TensorShape old_shape;
  TF_RETURN_IF_ERROR(GetNodeAttr(res_node->attrs(), "shape", &old_shape));
  int tensor_size = old_shape.dims();
  int new_part_shape;
  if (last_part) {
    new_part_shape = part_var_full_shape - part_var_full_shape /
                                               cur_partition_nums *
                                               (cur_partition_nums - 1);
  } else {
    new_part_shape = part_var_full_shape / cur_partition_nums;
  }

  new_shape = TensorShape({new_part_shape});
  if (tensor_size > 1) {
    for (int j = 1; j < tensor_size; ++j) {
      new_shape.AddDim(old_shape.dim_size(j));
    }
  }
  res_node->ClearAttr("shape");
  res_node->AddAttr("shape", new_shape);
  return Status::OK();
}

Status MakeConstInitializer(Graph* g, Node* new_init_op,
                            const TensorShape& new_shape,
                            const string& new_device_name) {
  Status s;
  Tensor init_tensor(DT_FLOAT, new_shape);
  init_tensor.flat<float>()(0) = 0;
  NodeDef init_value_def;
  TF_RETURN_IF_ERROR(NodeDefBuilder(new_init_op->name() + "/Const", "Const")
                         .Attr("dtype", DT_FLOAT)
                         .Attr("value", init_tensor)
                         .Device(new_device_name)
                         .Finalize(&init_value_def));
  Node* new_const_init = g->AddNode(init_value_def, &s);
  TF_RETURN_IF_ERROR(s);
  new_const_init->set_assigned_device_name(new_device_name);
  g->AddEdge(new_const_init, 0, new_init_op, 1);
  return s;
}

Status InitDynamicPartitionGraphMeta(const VarType& var_type, int part_num,
                                     PartIdToNodeMap& ev_node_vec,
                                     std::vector<Node*>& gather_node_vec,
                                     std::vector<Node*>& identity_node_vec,
                                     Node** dynamic_partition_node,
                                     Node** dynamic_stitch_node) {
  switch (var_type) {
    case VarType::EMBEDDING_VAR: {
      string gather_name = "KvResourceGather";
      for (int i = 0; i < part_num; ++i) {
        auto* ev_node = ev_node_vec[i];
        for (auto* o_node : ev_node->out_nodes()) {
          if (o_node->type_string() == gather_name) {
            gather_node_vec.push_back(o_node);
            const Edge* input_edge = nullptr;
            TF_RETURN_IF_ERROR(o_node->input_edge(1, &input_edge));
            if (input_edge->src()->type_string() == kDynamicPartition) {
              *dynamic_partition_node = input_edge->src();
            }

            for (auto* oo_node : o_node->out_nodes()) {
              if (oo_node->type_string() == kIdentityOp) {
                identity_node_vec.push_back(oo_node);
                for (auto* ooo_node : oo_node->out_nodes()) {
                  if (ooo_node->type_string() == kParaDynamicStitch) {
                    *dynamic_stitch_node = ooo_node;
                  }
                }
              }
            }
          }
        }
      }
      break;
    }
    case VarType::REF_VAR: {
      string gather_name = "GatherV2";
      for (int i = 0; i < part_num; ++i) {
        auto* ev_node = ev_node_vec[i];
        for (auto* o_node : ev_node->out_nodes()) {
          if (o_node->type_string() == kIdentityOp) {
            for (auto* oo_node : o_node->out_nodes()) {
              if (oo_node->type_string() == gather_name) {
                gather_node_vec.push_back(oo_node);
                identity_node_vec.push_back(oo_node);  // trick
                if (i == 0) {
                  const Edge* input_edge = nullptr;
                  TF_RETURN_IF_ERROR(oo_node->input_edge(1, &input_edge));
                  if (input_edge->src()->type_string() == kDynamicPartition) {
                    *dynamic_partition_node = input_edge->src();
                  }

                  for (auto* ooo_node : oo_node->out_nodes()) {
                    if (ooo_node->type_string() == kParaDynamicStitch) {
                      *dynamic_stitch_node = ooo_node;
                    }
                  }
                }
              }
            }
          }
        }
      }
      break;
    }
    case VarType::RESOURCE_VAR: {
      string gather_name = "ResourceGather";
      for (int i = 0; i < part_num; ++i) {
        auto* ev_node = ev_node_vec[i];
        for (auto* o_node : ev_node->out_nodes()) {
          if (o_node->type_string() == gather_name) {
            gather_node_vec.push_back(o_node);
            const Edge* input_edge = nullptr;
            TF_RETURN_IF_ERROR(o_node->input_edge(1, &input_edge));
            if (input_edge->src()->type_string() == kDynamicPartition) {
              *dynamic_partition_node = input_edge->src();
            }

            for (auto* oo_node : o_node->out_nodes()) {
              if (oo_node->type_string() == kIdentityOp) {
                identity_node_vec.push_back(oo_node);
                for (auto* ooo_node : oo_node->out_nodes()) {
                  if (ooo_node->type_string() == kParaDynamicStitch) {
                    *dynamic_stitch_node = ooo_node;
                  }
                }
              }
            }
          }
        }
      }
      break;
    }
    default:  // DENSE_LAYER_VAR
      return Status::OK();
  }

  if ((*dynamic_stitch_node == nullptr) ||
      (*dynamic_partition_node == nullptr)) {
    return errors::Internal(
        "dynamic_stitch_node or dynamic_partition_node is nullptr");
  }
  return Status::OK();
}

Status MakeElasticPartitionOp(const VarType& var_type,
                              const Node* dynamic_partition_node, int part_num,
                              const std::vector<Node*>& gather_node_vec,
                              Graph* g, Node** elastic_node,
                              std::unordered_set<Node*>& nodes_to_delete) {
  string partition_strategy;
  switch (var_type) {
    case VarType::EMBEDDING_VAR: {
      partition_strategy = "bucket";
      break;
    }
    case VarType::REF_VAR: {
      partition_strategy = "mod";
      break;
    }
    default:
      return Status::OK();
  }

  Status s;
  std::string node_name = dynamic_partition_node->name();
  DataType key_type;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(dynamic_partition_node->attrs(), "T", &key_type));
  int num_partitions;
  TF_RETURN_IF_ERROR(GetNodeAttr(dynamic_partition_node->attrs(),
                                 "num_partitions", &num_partitions));
  const Node* a_copy;
  TF_RETURN_IF_ERROR(dynamic_partition_node->input_node(0, &a_copy));
  const Node* b_copy;
  TF_RETURN_IF_ERROR(dynamic_partition_node->input_node(1, &b_copy));
  auto idx = node_name.find(kDynamicPartition);
  std::string pre_node_name = node_name.substr(0, idx - 1);
  NodeDef elastic_node_def;
  TF_RETURN_IF_ERROR(NodeDefBuilder(pre_node_name + "/ElasticPartition",
                                    ::des::kElasticPartition)
                         .Input(a_copy->name(), 0, a_copy->output_type(0))
                         .Input(b_copy->name(), 0, b_copy->output_type(0))
                         .Attr("num_partitions", part_num)
                         .Attr("TKey", key_type)
                         .Attr("partition_strategy", partition_strategy)
                         .Device(dynamic_partition_node->assigned_device_name())
                         .Finalize(&elastic_node_def));
  *elastic_node = g->AddNode(elastic_node_def, &s);
  TF_RETURN_IF_ERROR(s);
  (*elastic_node)
      ->set_assigned_device_name(
          dynamic_partition_node->assigned_device_name());

  const Edge* input_edge = nullptr;
  TF_RETURN_IF_ERROR(dynamic_partition_node->input_edge(1, &input_edge));
  for (auto* o_node : input_edge->src()->out_nodes()) {
    if (o_node->type_string() == kDynamicPartition) {
      const Edge* data_input_edge = nullptr;
      TF_RETURN_IF_ERROR(o_node->input_edge(0, &data_input_edge));
      if (data_input_edge->src()->type_string() != "Range") {  // ID
        // Input
        g->AddEdge(data_input_edge->src(), data_input_edge->src_output(),
                   *elastic_node, 0);
        nodes_to_delete.insert(o_node);

      } else {  // Indices
        // Input
        g->AddEdge(data_input_edge->src(), data_input_edge->src_output(),
                   *elastic_node, 1);
        nodes_to_delete.insert(o_node);
      }
    }
  }
  if (num_partitions > part_num) {
    for (int i = 0; i < gather_node_vec.size(); ++i) {
      TF_RETURN_IF_ERROR(
          g->UpdateEdge(*elastic_node, i, gather_node_vec[i], 1));
    }
  } else {
    for (int i = 0; i < gather_node_vec.size(); ++i) {
      if (i < num_partitions) {
        TF_RETURN_IF_ERROR(
            g->UpdateEdge(*elastic_node, i, gather_node_vec[i], 1));
      } else {
        g->AddEdge(*elastic_node, i, gather_node_vec[i], 1);
      }
    }
  }

  nodes_to_delete.insert(input_edge->src());
  return s;
}

Status MakeDynamicPartitionOp(const VarType& var_type,
                              const Node* dynamic_partition_node, int part_num,
                              const std::vector<Node*>& gather_node_vec,
                              Graph* g, Node** data_dp_node,
                              Node** indices_dp_node,
                              std::unordered_set<Node*>& nodes_to_delete) {
  Status s;
  std::string node_name = dynamic_partition_node->name();
  int num_partitions;
  TF_RETURN_IF_ERROR(GetNodeAttr(dynamic_partition_node->attrs(),
                                 "num_partitions", &num_partitions));
  const Edge* input_edge = nullptr;
  TF_RETURN_IF_ERROR(dynamic_partition_node->input_edge(1, &input_edge));
  switch (var_type) {
    case VarType::EMBEDDING_VAR: {
      Node* floormod_node = nullptr;
      TF_RETURN_IF_ERROR(input_edge->src()->input_node(0, &floormod_node));
      Node* divisor_node = nullptr;
      TF_RETURN_IF_ERROR(floormod_node->input_node(1, &divisor_node));
      Tensor new_part_nums(DT_INT64, TensorShape({}));
      new_part_nums.flat<int64>()(0) = part_num;
      divisor_node->ClearAttr("value");
      divisor_node->AddAttr("value", new_part_nums);
    }
    case VarType::REF_VAR: {
      Node* maximum_node = nullptr;
      TF_RETURN_IF_ERROR(input_edge->src()->input_node(0, &maximum_node));
      Node* floormod_node = nullptr;
      TF_RETURN_IF_ERROR(maximum_node->input_node(0, &floormod_node));
      Node* add_node = nullptr;
      TF_RETURN_IF_ERROR(floormod_node->input_node(1, &add_node));
      Node* fl_node = nullptr;
      TF_RETURN_IF_ERROR(add_node->input_node(0, &fl_node));
      Node* const_node = nullptr;
      TF_RETURN_IF_ERROR(fl_node->input_node(1, &const_node));
      Tensor new_part_nums(DT_INT64, TensorShape({}));
      new_part_nums.flat<int64>()(0) = part_num;
      const_node->ClearAttr("value");
      const_node->AddAttr("value", new_part_nums);

      Node* floormod_node_1 = nullptr;
      TF_RETURN_IF_ERROR(maximum_node->input_node(1, &floormod_node_1));
      Node* sub_node = nullptr;
      TF_RETURN_IF_ERROR(floormod_node_1->input_node(0, &sub_node));
      Node* mod_node = nullptr;
      TF_RETURN_IF_ERROR(sub_node->input_node(1, &mod_node));
      Node* const_node_1 = nullptr;
      TF_RETURN_IF_ERROR(mod_node->input_node(1, &const_node_1));
      const_node_1->ClearAttr("value");
      const_node_1->AddAttr("value", new_part_nums);
    }
  }

  for (auto* o_node : input_edge->src()->out_nodes()) {
    if (o_node->type_string() == kDynamicPartition) {
      const Edge* data_input_edge = nullptr;
      TF_RETURN_IF_ERROR(o_node->input_edge(0, &data_input_edge));
      if (data_input_edge->src()->type_string() != "Range") {  // ID
        // Input
        *data_dp_node = CopyNode(g, dynamic_partition_node,
                                 dynamic_partition_node->assigned_device_name(),
                                 0, "DES/" + dynamic_partition_node->name());
        (*data_dp_node)->ClearAttr("num_partitions");
        (*data_dp_node)->AddAttr("num_partitions", part_num);
        g->AddEdge(data_input_edge->src(), data_input_edge->src_output(),
                   *data_dp_node, 0);
        g->AddEdge(input_edge->src(), input_edge->src_output(), *data_dp_node,
                   1);
        nodes_to_delete.insert(o_node);

      } else {  // Indices
        *indices_dp_node = CopyNode(g, o_node, o_node->assigned_device_name(),
                                    0, "DES/" + o_node->name());
        (*indices_dp_node)->ClearAttr("num_partitions");
        (*indices_dp_node)->AddAttr("num_partitions", part_num);
        g->AddEdge(data_input_edge->src(), data_input_edge->src_output(),
                   *indices_dp_node, 0);
        g->AddEdge(input_edge->src(), input_edge->src_output(),
                   *indices_dp_node, 1);
        nodes_to_delete.insert(o_node);
      }
    }
  }
  if (num_partitions > part_num) {
    for (int i = 0; i < gather_node_vec.size(); ++i) {
      TF_RETURN_IF_ERROR(
          g->UpdateEdge(*data_dp_node, i, gather_node_vec[i], 1));
    }
  } else {
    for (int i = 0; i < gather_node_vec.size(); ++i) {
      if (i < num_partitions) {
        TF_RETURN_IF_ERROR(
            g->UpdateEdge(*data_dp_node, i, gather_node_vec[i], 1));
      } else {
        g->AddEdge(*data_dp_node, i, gather_node_vec[i], 1);
      }
    }
  }

  return s;
}

void MakeDynamicStitchOp(Node* dynamic_stitch_node, int part_num,
                         const std::vector<Node*>& identity_node_vec, Graph* g,
                         Node* indices_dp_node, Node** new_dynamic_stitch_node,
                         std::unordered_set<Node*>& nodes_to_delete) {
  *new_dynamic_stitch_node = CopyNode(
      g, dynamic_stitch_node, dynamic_stitch_node->assigned_device_name(), 0);
  (*new_dynamic_stitch_node)->ClearAttr("N");
  (*new_dynamic_stitch_node)->AddAttr("N", part_num);
  nodes_to_delete.insert(dynamic_stitch_node);
  for (auto* o_edge : dynamic_stitch_node->out_edges()) {
    g->AddEdge(*new_dynamic_stitch_node, o_edge->src_output(), o_edge->dst(),
               o_edge->dst_input());
  }

  for (int i = 0; i < identity_node_vec.size(); ++i) {
    g->AddEdge(indices_dp_node, i, *new_dynamic_stitch_node, i);
    g->AddEdge(identity_node_vec[i], 0, *new_dynamic_stitch_node, part_num + i);
  }
}

Status UpdateOldBackWardGraph(const VarType& var_type,
                              const Node* variable_node, Node* data_dp_node,
                              Node* indices_dp_node,
                              const std::vector<std::string>& opt_ev_names,
                              Graph* g, int i, int part_num) {
  for (auto* node : variable_node->out_nodes()) {
    if (IsApplyNode(var_type, node)) {
      for (int j = node->num_inputs() - 1; j > opt_ev_names.size(); --j) {
        Node* i_node;
        TF_RETURN_IF_ERROR(node->input_node(j, &i_node));
        if (i_node->IsUnique()) {
          Node* unique_indices_node;
          TF_RETURN_IF_ERROR(i_node->input_node(0, &unique_indices_node));
          TF_RETURN_IF_ERROR(
              g->UpdateEdge(data_dp_node, i, unique_indices_node, 0));
          Node* expand_dim_node;
          TF_RETURN_IF_ERROR(
              unique_indices_node->input_node(1, &expand_dim_node));
          Node* size_node;
          TF_RETURN_IF_ERROR(expand_dim_node->input_node(0, &size_node));
          TF_RETURN_IF_ERROR(g->UpdateEdge(data_dp_node, i, size_node, 0));
        }

        if (i_node->type_string() == "UnsortedSegmentSum") {
          Node* reshape_node;
          TF_RETURN_IF_ERROR(i_node->input_node(0, &reshape_node));
          Node* control_dependency_node;
          TF_RETURN_IF_ERROR(
              reshape_node->input_node(0, &control_dependency_node));
          Node* gather_node;
          TF_RETURN_IF_ERROR(
              control_dependency_node->input_node(0, &gather_node));
          TF_RETURN_IF_ERROR(g->UpdateEdge(indices_dp_node, i, gather_node, 1));
        }
      }
    }
  }
  return Status::OK();
}

Status MakeSparseApplyOp(
    Graph* g, const Node* old_apply_node, Node* data_dp_node,
    Node* indices_dp_node, Node* cur_noop_node,
    const std::string& primary_ev_name,
    const std::vector<std::string>& opt_ev_names, const string& new_device_name,
    std::unordered_map<std::string, PartIdToNodeMap>& node_to_origin_map, int i,
    int cur_partition_nums) {
  Node* new_apply_node = CopyNode(g, old_apply_node, new_device_name, i);
  g->AddControlEdge(new_apply_node, cur_noop_node);
  g->AddEdge(node_to_origin_map[primary_ev_name][i], 0, new_apply_node, 0);
  for (int j = 0; j < opt_ev_names.size(); ++j) {
    g->AddEdge(node_to_origin_map[opt_ev_names[j]][i], 0, new_apply_node,
               j + 1);
  }
  Node* new_unique = nullptr;
  Node* new_expand_dims = nullptr;
  for (int j = old_apply_node->num_inputs() - 1; j > opt_ev_names.size(); --j) {
    Node* i_node;
    TF_RETURN_IF_ERROR(old_apply_node->input_node(j, &i_node));
    if (i_node->IsUnique()) {
      new_unique = CopyNode(g, i_node, new_device_name, i);
      g->AddEdge(new_unique, 0, new_apply_node, j);
      // unique INPUT 0
      Node* reshape_id;
      TF_RETURN_IF_ERROR(i_node->input_node(0, &reshape_id));
      Node* new_reshape_id =
          CopyNode(g, reshape_id, reshape_id->assigned_device_name(), i);
      g->AddEdge(new_reshape_id, 0, new_unique, 0);

      for (auto* o_node : reshape_id->out_nodes()) {
        if (o_node->type_string() == "RecordSparseIndices") {
          Node* new_record_sparse = CopyNode(g, o_node, new_device_name, i);
          g->AddEdge(new_reshape_id, 0, new_record_sparse, 0);
          g->AddControlEdge(new_record_sparse, cur_noop_node);
          for (auto* o_edge : o_node->out_edges()) {
            if (o_edge->IsControlEdge() &&
                o_edge->dst()->type_string() != "NoOp") {
              g->AddControlEdge(new_record_sparse, o_edge->dst());
              g->AddControlEdge(new_apply_node, o_edge->dst());
            }
          }
        }
      }

      // Reshape INPUT
      g->AddEdge(data_dp_node, i, new_reshape_id, 0);

      Node* expand_dims;
      TF_RETURN_IF_ERROR(reshape_id->input_node(1, &expand_dims));
      new_expand_dims =
          CopyNode(g, expand_dims, expand_dims->assigned_device_name(), i);
      g->AddEdge(new_expand_dims, 0, new_reshape_id, 1);

      // expand dims INPUT
      Node* expand_dims_size;
      TF_RETURN_IF_ERROR(expand_dims->input_node(0, &expand_dims_size));
      Node* new_expand_dims_size = CopyNode(
          g, expand_dims_size, expand_dims_size->assigned_device_name(), i);
      g->AddEdge(new_expand_dims_size, 0, new_expand_dims, 0);
      g->AddEdge(data_dp_node, i, new_expand_dims_size, 0);

      Node* expand_dims_dim;
      TF_RETURN_IF_ERROR(expand_dims->input_node(1, &expand_dims_dim));
      Node* new_expand_dims_dim = CopyNode(
          g, expand_dims_dim, expand_dims_dim->assigned_device_name(), i);
      g->AddEdge(new_expand_dims_dim, 0, new_expand_dims, 1);

    } else if (i_node->type_string() == "UnsortedSegmentSum") {
      /*
        control_dependency          Reshape ->
        ExpandDims
        ElasticPartition: ID -> Unique: idx ->   UnsortedSegmentSum
                                strided_slice ->
                  |
                  v
                  SparseRecordIndices
      */
      Node* new_unsorted_segment = CopyNode(g, i_node, new_device_name, i);
      g->AddEdge(new_unsorted_segment, 0, new_apply_node, j);
      // Input 0
      {
        Node* reshape;
        TF_RETURN_IF_ERROR(i_node->input_node(0, &reshape));
        Node* new_reshape =
            CopyNode(g, reshape, reshape->assigned_device_name(), i);
        g->AddEdge(new_reshape, 0, new_unsorted_segment, 0);
        // Reshape INPUT 0
        Node* control_denpency;
        TF_RETURN_IF_ERROR(reshape->input_node(0, &control_denpency));
        Node* new_control_denpency = CopyNode(
            g, control_denpency, control_denpency->assigned_device_name(), i);
        g->AddEdge(new_control_denpency, 0, new_reshape, 0);

        for (auto* i_edge : control_denpency->in_edges()) {
          if (i_edge->IsControlEdge()) {
            g->AddControlEdge(i_edge->src(), new_control_denpency);
          }
        }

        // control_dependency INPUT 0
        Node* gather_1;
        TF_RETURN_IF_ERROR(control_denpency->input_node(0, &gather_1));
        Node* new_gather_1 =
            CopyNode(g, gather_1, gather_1->assigned_device_name(), i);
        g->AddEdge(new_gather_1, 0, new_control_denpency, 0);
        for (auto* o_edge : gather_1->out_edges()) {
          if (o_edge->IsControlEdge()) {
            g->AddControlEdge(new_gather_1, o_edge->dst());
          }
        }

        Node* reshape_1;
        TF_RETURN_IF_ERROR(gather_1->input_node(0, &reshape_1));
        g->AddEdge(reshape_1, 0, new_gather_1, 0);

        // gather_1 INPUT1
        g->AddEdge(indices_dp_node, i /*idx*/, new_gather_1, 1);
        // gather_1 INPUT2
        Node* axis_1;
        TF_RETURN_IF_ERROR(gather_1->input_node(2, &axis_1));
        Node* new_axis_1 =
            CopyNode(g, axis_1, axis_1->assigned_device_name(), i);
        g->AddEdge(new_axis_1, 0, new_gather_1, 2);

        // Reshape INPUT 1
        Node* concat;
        TF_RETURN_IF_ERROR(reshape->input_node(1, &concat));
        Node* new_concat =
            CopyNode(g, concat, concat->assigned_device_name(), i);
        g->AddEdge(new_concat, 0, new_reshape, 1);

        // concat INPUT 0
        g->AddEdge(new_expand_dims, 0, new_concat, 0);

        // concat INPUT 1
        Node* strided_slice;
        TF_RETURN_IF_ERROR(concat->input_node(1, &strided_slice));
        Node* new_strided_slice = CopyNode(
            g, strided_slice, strided_slice->assigned_device_name(), i);
        g->AddEdge(new_strided_slice, 0, new_concat, 1);  // Const shape

        for (int k = 0; k < strided_slice->num_inputs(); ++k) {
          Node* partial_strided_slice;
          TF_RETURN_IF_ERROR(
              strided_slice->input_node(k, &partial_strided_slice));
          Node* new_node =
              CopyNode(g, partial_strided_slice,
                       partial_strided_slice->assigned_device_name(), i);
          g->AddEdge(new_node, 0, new_strided_slice, k);
        }

        // concat INPUT 2
        Node* axis;
        TF_RETURN_IF_ERROR(concat->input_node(2, &axis));
        Node* new_axis = CopyNode(g, axis, axis->assigned_device_name(), i);
        g->AddEdge(new_axis, 0, new_concat, 2);
      }

      // Input 1
      g->AddEdge(new_unique, 1 /*idx*/, new_unsorted_segment, 1);
      // Input 2
      {
        Node* strided_slice;
        TF_RETURN_IF_ERROR(i_node->input_node(2, &strided_slice));
        Node* new_strided_slice =
            CopyNode(g, strided_slice, new_device_name, i);
        g->AddEdge(new_strided_slice, 0, new_unsorted_segment, 2);

        Node* shape;
        TF_RETURN_IF_ERROR(strided_slice->input_node(0, &shape));
        Node* new_shape = CopyNode(g, shape, new_device_name, i);
        g->AddEdge(new_unique, 0, new_shape, 0);
        g->AddEdge(new_shape, 0, new_strided_slice, 0);

        for (int k = 1; k < strided_slice->num_inputs(); ++k) {
          Node* partial_strided_slice;
          TF_RETURN_IF_ERROR(
              strided_slice->input_node(k, &partial_strided_slice));
          Node* new_node =
              CopyNode(g, partial_strided_slice, new_device_name, i);
          g->AddEdge(new_node, 0, new_strided_slice, k);
        }
      }
    } else {
      g->AddEdge(i_node, 0, new_apply_node, j);
    }
  }
  return Status::OK();
}

Status UpdateOldDenseBackWardGraph(VarType var_type, Graph* g,
                                   Node* dense_variable_node,
                                   int part_var_full_shape, int i,
                                   int cur_partition_nums,
                                   int ev_partition_num) {
  for (auto* node : dense_variable_node->out_nodes()) {
    if (IsApplyNode(var_type, node)) {
      Node* i_node;
      TF_RETURN_IF_ERROR(node->input_node(node->num_inputs() - 1, &i_node));
      if (i_node->type_string() == "Identity") {
        Node* concat_grad_node;
        TF_RETURN_IF_ERROR(i_node->input_node(0, &concat_grad_node));
        Node* shape_node;
        TF_RETURN_IF_ERROR(concat_grad_node->input_node(2, &shape_node));
        Tensor old_shape_tensor;
        TF_RETURN_IF_ERROR(
            GetNodeAttr(shape_node->attrs(), "value", &old_shape_tensor));
        int tensor_size = old_shape_tensor.NumElements();
        int partition_shape = part_var_full_shape / cur_partition_nums;
        int remainder = part_var_full_shape % cur_partition_nums;
        if (remainder > i) {
          partition_shape += 1;
        }
        int new_part_shape = partition_shape;
        Tensor shape_tensor;
        TensorProto tensor_shape_proto;
        tensor_shape_proto.set_dtype(DT_INT32);
        TensorShape({tensor_size})
            .AsProto(tensor_shape_proto.mutable_tensor_shape());
        tensor_shape_proto.add_int_val(new_part_shape);
        if (tensor_size > 1) {
          for (int j = 1; j < tensor_size; ++j) {
            tensor_shape_proto.add_int_val(old_shape_tensor.flat<int>()(j));
          }
        }
        bool ret = shape_tensor.FromProto(tensor_shape_proto);
        if (!ret) return errors::Internal("shape tensor init error");
        shape_node->ClearAttr("value");
        shape_node->AddAttr("value", shape_tensor);

        const Edge* concat_offset_edge;
        TF_RETURN_IF_ERROR(
            concat_grad_node->input_edge(1, &concat_offset_edge));
        const Edge* target_edge = nullptr;
        for (auto* o_edge : shape_node->out_edges()) {
          if (o_edge->dst() == concat_offset_edge->src()) {
            target_edge = o_edge;
          }
        }
        int part_num;
        TF_RETURN_IF_ERROR(
            GetNodeAttr(concat_offset_edge->src()->attrs(), "N", &part_num));

        if (part_num != cur_partition_nums) {
          concat_offset_edge->src()->ClearAttr("N");
          concat_offset_edge->src()->AddAttr("N", cur_partition_nums);
        }

        g->RemoveEdge(target_edge);
        g->AddEdge(shape_node, 0, concat_offset_edge->src(), i + 1);
        // concat offset grad
        if (concat_offset_edge->src_output() != i) {
          TF_RETURN_IF_ERROR(
              g->UpdateEdge(concat_offset_edge->src(), i, concat_grad_node, 1));
        }
      }
    }
  }
  return Status::OK();
}

Status MakeNoOp(Graph* g, Node** cur_noop_node, Node* cur_apply_node,
                const string& new_device_name, std::vector<Node*>& no_op_vec,
                int i) {
  Status s;
  if (*cur_noop_node == nullptr) {
    for (auto* node : g->op_nodes()) {
      if (node->name() == "head/Optimizer/update/NoOp_" + std::to_string(i)) {
        *cur_noop_node = node;
        no_op_vec[i] = node;
        return s;
      }
    }

    NodeDef noop_def;
    TF_RETURN_IF_ERROR(
        NodeDefBuilder("head/Optimizer/update/NoOp_" + std::to_string(i),
                       "NoOp")
            .Device(new_device_name)
            .Finalize(&noop_def));
    Node* no_node = g->AddNode(noop_def, &s);
    TF_RETURN_IF_ERROR(s);
    for (auto* edge : cur_apply_node->out_edges()) {
      for (auto* o_edge : edge->dst()->out_edges()) {
        if (o_edge->IsControlEdge()) {
          g->AddControlEdge(no_node, o_edge->dst());
        }
      }
    }
    *cur_noop_node = no_node;
    no_op_vec[i] = no_node;
  }
  return s;
}

Status MakeApplyOp(
    Graph* g, const Node* old_apply_node, Node* cur_noop_node,
    const std::string& primary_ev_name,
    const std::vector<std::string>& opt_ev_names,
    const std::string& new_device_name,
    std::unordered_map<std::string, PartIdToNodeMap>& node_to_origin_map, int i,
    int cur_partition_nums, int part_var_full_shape) {
  Node* new_apply_node = CopyNode(g, old_apply_node, new_device_name, i);
  g->AddControlEdge(new_apply_node, cur_noop_node);
  g->AddEdge(node_to_origin_map[primary_ev_name][i], 0, new_apply_node, 0);
  for (int j = 0; j < opt_ev_names.size(); ++j) {
    g->AddEdge(node_to_origin_map[opt_ev_names[j]][i], 0, new_apply_node,
               j + 1);
  }

  Node* i_node;
  TF_RETURN_IF_ERROR(
      old_apply_node->input_node(old_apply_node->num_inputs() - 1, &i_node));
  if (i_node->type_string() == "Identity") {
    Node* new_grad_node = CopyNode(g, i_node, new_device_name, i);
    g->AddEdge(new_grad_node, 0, new_apply_node,
               old_apply_node->num_inputs() - 1);

    Node* concat_grad_node;
    TF_RETURN_IF_ERROR(i_node->input_node(0, &concat_grad_node));
    Node* new_concat_grad_node = CopyNode(
        g, concat_grad_node, concat_grad_node->assigned_device_name(), i);
    g->AddEdge(new_concat_grad_node, 0, new_grad_node, 0);

    Node* prev_grad_node;
    TF_RETURN_IF_ERROR(concat_grad_node->input_node(0, &prev_grad_node));
    g->AddEdge(prev_grad_node, 0, new_concat_grad_node, 0);

    Node* concat_offset_node;
    TF_RETURN_IF_ERROR(concat_grad_node->input_node(1, &concat_offset_node));
    int part_num;
    TF_RETURN_IF_ERROR(
        GetNodeAttr(concat_offset_node->attrs(), "N", &part_num));

    if (part_num != cur_partition_nums) {
      concat_offset_node->ClearAttr("N");
      concat_offset_node->AddAttr("N", cur_partition_nums);
    }

    g->AddEdge(concat_offset_node, i, new_concat_grad_node, 1);
    Node* shape_node;
    TF_RETURN_IF_ERROR(concat_offset_node->input_node(1, &shape_node));
    Tensor old_shape_tensor;
    TF_RETURN_IF_ERROR(
        GetNodeAttr(shape_node->attrs(), "value", &old_shape_tensor));
    int tensor_size = old_shape_tensor.NumElements();
    Node* new_shape_node =
        CopyNode(g, shape_node, shape_node->assigned_device_name(), i);
    int new_part_shape = part_var_full_shape / cur_partition_nums;
    if (part_var_full_shape % cur_partition_nums > i) {
      new_part_shape += 1;
    }
    Tensor shape_tensor;
    TensorProto tensor_shape_proto;
    tensor_shape_proto.set_dtype(DT_INT32);
    TensorShape({tensor_size})
        .AsProto(tensor_shape_proto.mutable_tensor_shape());
    tensor_shape_proto.add_int_val(new_part_shape);
    if (tensor_size > 1) {
      for (int j = 1; j < tensor_size; ++j) {
        tensor_shape_proto.add_int_val(old_shape_tensor.flat<int>()(j));
      }
    }
    bool ret = shape_tensor.FromProto(tensor_shape_proto);
    if (!ret) return errors::Internal("shape tensor init error");
    new_shape_node->ClearAttr("value");
    new_shape_node->AddAttr("value", shape_tensor);
    g->AddEdge(new_shape_node, 0, concat_offset_node, i + 1);
    g->AddEdge(new_shape_node, 0, new_concat_grad_node, 2);
    // TODO grad value size
    for (auto* i_edge : i_node->in_edges()) {
      if (i_edge->IsControlEdge()) {
        Node* control_node = i_edge->src();
        g->AddControlEdge(new_concat_grad_node, control_node);
        g->AddControlEdge(control_node, new_grad_node);
      }
    }
  }

  for (int j = old_apply_node->num_inputs() - 2; j > opt_ev_names.size(); --j) {
    Node* i_node;
    TF_RETURN_IF_ERROR(old_apply_node->input_node(j, &i_node));
    g->AddEdge(i_node, 0, new_apply_node, j);
  }
  return Status::OK();
}

Status DeleteSparseBackWardGraph(VarType var_type, Node* cur_ev_node,
                                 const std::vector<string>& opt_ev_names,
                                 std::unordered_set<Node*>& nodes_to_delete) {
  for (auto* node : cur_ev_node->out_nodes()) {
    if (IsApplyNode(var_type, node)) {
      nodes_to_delete.insert(node);
      for (int j = node->num_inputs() - 1; j > opt_ev_names.size(); --j) {
        Node* i_node;
        TF_RETURN_IF_ERROR(node->input_node(j, &i_node));
        if (i_node->IsUnique()) {
          nodes_to_delete.insert(i_node);
          // unique INPUT 0
          Node* reshape_id;
          TF_RETURN_IF_ERROR(i_node->input_node(0, &reshape_id));
          nodes_to_delete.insert(reshape_id);

          for (auto* o_node : reshape_id->out_nodes()) {
            if (o_node->type_string() == "RecordSparseIndices") {
              nodes_to_delete.insert(o_node);
            }
          }

          Node* expand_dims;
          TF_RETURN_IF_ERROR(reshape_id->input_node(1, &expand_dims));
          nodes_to_delete.insert(expand_dims);

          // expand dims INPUT
          Node* expand_dims_size;
          TF_RETURN_IF_ERROR(expand_dims->input_node(0, &expand_dims_size));
          nodes_to_delete.insert(expand_dims_size);

          Node* expand_dims_dim;
          TF_RETURN_IF_ERROR(expand_dims->input_node(1, &expand_dims_dim));
          nodes_to_delete.insert(expand_dims_dim);

        } else if (i_node->type_string() == "UnsortedSegmentSum") {
          /*
            control_dependency          Reshape ->
            ExpandDims
            ElasticPartition: ID -> Unique: idx ->   UnsortedSegmentSum
                                    strided_slice ->
                      |
                      v
                      SparseRecordIndices
          */
          nodes_to_delete.insert(i_node);
          // Input 0
          {
            Node* reshape;
            TF_RETURN_IF_ERROR(i_node->input_node(0, &reshape));
            nodes_to_delete.insert(reshape);
            // Reshape INPUT 0
            Node* control_denpency;
            TF_RETURN_IF_ERROR(reshape->input_node(0, &control_denpency));
            nodes_to_delete.insert(control_denpency);

            // control_dependency INPUT 0
            Node* gather_1;
            TF_RETURN_IF_ERROR(control_denpency->input_node(0, &gather_1));
            nodes_to_delete.insert(gather_1);

            // gather_1 INPUT2
            Node* axis_1;
            TF_RETURN_IF_ERROR(gather_1->input_node(2, &axis_1));
            nodes_to_delete.insert(axis_1);

            // Reshape INPUT 1
            Node* concat;
            TF_RETURN_IF_ERROR(reshape->input_node(1, &concat));
            nodes_to_delete.insert(concat);

            // concat INPUT 1
            Node* strided_slice;
            TF_RETURN_IF_ERROR(concat->input_node(1, &strided_slice));
            nodes_to_delete.insert(strided_slice);

            for (int k = 0; k < strided_slice->num_inputs(); ++k) {
              Node* partial_strided_slice;
              TF_RETURN_IF_ERROR(
                  strided_slice->input_node(k, &partial_strided_slice));
              nodes_to_delete.insert(partial_strided_slice);
            }

            // concat INPUT 2
            Node* axis;
            TF_RETURN_IF_ERROR(concat->input_node(2, &axis));
            nodes_to_delete.insert(axis);
          }

          // Input 2
          {
            Node* strided_slice;
            TF_RETURN_IF_ERROR(i_node->input_node(2, &strided_slice));
            nodes_to_delete.insert(strided_slice);

            Node* shape;
            TF_RETURN_IF_ERROR(strided_slice->input_node(0, &shape));
            nodes_to_delete.insert(shape);

            for (int k = 1; k < strided_slice->num_inputs(); ++k) {
              Node* partial_strided_slice;
              TF_RETURN_IF_ERROR(
                  strided_slice->input_node(k, &partial_strided_slice));
              nodes_to_delete.insert(partial_strided_slice);
            }
          }
        }
      }
    }
  }
  return Status::OK();
}

Status DeleteDenseBackWardGraph(VarType var_type, Graph* g, Node* cur_var_node,
                                int cur_partition_nums,
                                std::unordered_set<Node*>& nodes_to_delete) {
  for (auto* node : cur_var_node->out_nodes()) {
    if (IsApplyNode(var_type, node)) {
      nodes_to_delete.insert(node);

      Node* i_node;
      TF_RETURN_IF_ERROR(node->input_node(node->num_inputs() - 1, &i_node));
      if (i_node->type_string() == "Identity") {
        nodes_to_delete.insert(i_node);

        Node* concat_grad_node;
        TF_RETURN_IF_ERROR(i_node->input_node(0, &concat_grad_node));
        nodes_to_delete.insert(concat_grad_node);

        Node* prev_grad_node;
        TF_RETURN_IF_ERROR(concat_grad_node->input_node(0, &prev_grad_node));

        Node* shape_node;
        TF_RETURN_IF_ERROR(concat_grad_node->input_node(2, &shape_node));
        nodes_to_delete.insert(shape_node);
      }
    }
  }
  return Status::OK();
}

void InitPartVariableShape(
    std::unordered_map<std::string, PartitionedVariable>&
        primary_node_metas_map,
    std::unordered_map<string, std::vector<int64>>& partitioned_variable_shape,
    std::unordered_map<string, string>& opt_to_primary_map) {
  for (auto& it : primary_node_metas_map) {
    PartitionedVariable partition_var = it.second;
    for (auto& map_it : partition_var.node_map) {
      partitioned_variable_shape.emplace(std::pair<string, std::vector<int64>>(
          map_it.first, std::vector<int64>{}));
    }
    for (auto& opt_name : partition_var.sorted_opt_ev_names) {
      opt_to_primary_map.emplace(
          std::pair<string, string>(opt_name, partition_var.variable_prefix));
    }
  }
}

Status ScalingSaverSaverNodeUtil(
    Graph* g, Node* ori_save_node, int i,
    std::unordered_map<std::string, PartitionedVariable>&
        primary_node_metas_map,
    string& assigned_device_name, bool& has_ev,
    std::vector<Node*>& kv_lookup_resource_node_vec,
    std::vector<string>& ev_names_vec, std::vector<DataType>& key_data_types,
    std::vector<string>& tensor_names_vec,
    std::vector<Node*>& restore_tensor_vec, std::vector<Node*>& tensor_vec,
    std::vector<NodeDefBuilder::NodeOut>& tensors_input,
    std::vector<DataType>& n_dtypes) {
  Status s;
  for (auto& it : primary_node_metas_map) {
    PartitionedVariable partition_var = it.second;
    for (auto& map_it : partition_var.node_map) {
      auto ev_node = map_it.second[i];
      bool is_ev = it.second.var_type == VarType::EMBEDDING_VAR;
      if (is_ev) {
        has_ev = true;
        DataType key_type, value_type;
        TF_RETURN_IF_ERROR(GetNodeAttr(ev_node->attrs(), "Tkeys", &key_type));
        TF_RETURN_IF_ERROR(GetNodeAttr(ev_node->attrs(), "dtype", &value_type));
        NodeDef kv_lookup_resource_node_def;
        TF_RETURN_IF_ERROR(
            NodeDefBuilder(ev_node->name() + "/KvResourceLookupResource",
                           "KvResourceLookupResource")
                .Input(ev_node->name(), 0, ev_node->output_type(0))
                .Attr("Tkeys", key_type)
                .Attr("dtype", value_type)
                .Device(assigned_device_name)
                .Finalize(&kv_lookup_resource_node_def));
        Node* kv_lookup_resource_node =
            g->AddNode(kv_lookup_resource_node_def, &s);
        TF_RETURN_IF_ERROR(s);
        kv_lookup_resource_node->set_assigned_device_name(assigned_device_name);
        kv_lookup_resource_node_vec.emplace_back(kv_lookup_resource_node);
        ev_names_vec.emplace_back(ev_node->name());
        key_data_types.emplace_back(key_type);
      } else {
        tensor_names_vec.emplace_back(map_it.first);
        restore_tensor_vec.emplace_back(ev_node);
        for (auto* o_node : ev_node->out_nodes()) {
          if (o_node->type_string() == "Identity") {
            tensor_vec.emplace_back(o_node);
          } else if (o_node->type_string() == "ReadVariableOp") {
            for (auto* oo_node : o_node->out_nodes()) {
              if (oo_node->type_string() == "Identity") {
                for (auto* ooo_node : oo_node->out_nodes()) {
                  if (ooo_node->type_string() == "Identity") {
                    tensor_vec.emplace_back(ooo_node);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  if (has_ev) {
    // global_step
    Node* global_step_node;
    TF_RETURN_IF_ERROR(ori_save_node->input_node(5, &global_step_node));
    tensors_input.emplace_back(NodeDefBuilder::NodeOut(
        global_step_node->name(), 0, global_step_node->output_type(0)));
    n_dtypes.emplace_back(DT_INT64);
  }

  for (auto& tensor : tensor_vec) {
    tensors_input.emplace_back(
        NodeDefBuilder::NodeOut(tensor->name(), 0, tensor->output_type(0)));
    DataType t_type;
    TF_RETURN_IF_ERROR(GetNodeAttr(tensor->attrs(), "T", &t_type));
    n_dtypes.emplace_back(t_type);
  }
  return s;
}

Status AddNewSaverGraph(
    Graph* g, bool& has_ev, Node** new_sharded_filename,
    std::vector<string>& tensor_names_vec, std::vector<DataType>& n_dtypes,
    std::unordered_map<std::string, PartitionedVariable>&
        primary_node_metas_map,
    std::unordered_map<string, std::vector<int64>>& variable_shape,
    std::vector<Node*>& restore_tensor_vec, std::string& assigned_device_name,
    std::vector<Node*>& save_node_vec, int i, int cur_partition_nums) {
  Node* ori_save_node = save_node_vec[0];
  std::vector<Node*> tensor_vec;
  std::vector<Node*> kv_lookup_resource_node_vec;
  std::vector<DataType> ev_dtypes;
  std::vector<string> ev_names_vec;
  std::vector<DataType> key_data_types;
  std::vector<NodeDefBuilder::NodeOut> tensors_input;
  TF_RETURN_IF_ERROR(ScalingSaverSaverNodeUtil(
      g, ori_save_node, i, primary_node_metas_map, assigned_device_name, has_ev,
      kv_lookup_resource_node_vec, ev_names_vec, key_data_types,
      tensor_names_vec, restore_tensor_vec, tensor_vec, tensors_input,
      n_dtypes));
  Status s;
  Node* tensor_name_node;
  Node* shape_slice_node;
  Node* ev_name_node;
  Node* kv_lookup_resource_node;

  {
    Node* sharded_filename;
    TF_RETURN_IF_ERROR(ori_save_node->input_node(0, &sharded_filename));
    *new_sharded_filename =
        CopyNode(g, sharded_filename, assigned_device_name, i);

    Node* prefix_name;
    TF_RETURN_IF_ERROR(sharded_filename->input_node(0, &prefix_name));
    g->AddEdge(prefix_name, 0, *new_sharded_filename, 0);

    Node* num_shards;
    TF_RETURN_IF_ERROR(sharded_filename->input_node(2, &num_shards));
    Tensor new_tensor_nums(DT_INT32, TensorShape({}));
    new_tensor_nums.flat<int32>()(0) = cur_partition_nums;
    num_shards->ClearAttr("value");
    num_shards->AddAttr("value", new_tensor_nums);
    g->AddEdge(num_shards, 0, *new_sharded_filename, 2);

    Node* id_shards;
    TF_RETURN_IF_ERROR(sharded_filename->input_node(1, &id_shards));
    Node* new_id_shards = CopyNode(g, id_shards, assigned_device_name, i);
    Tensor new_tensor_ids(DT_INT32, TensorShape({}));
    new_tensor_ids.flat<int32>()(0) = i;
    new_id_shards->ClearAttr("value");
    new_id_shards->AddAttr("value", new_tensor_ids);
    g->AddEdge(new_id_shards, 0, *new_sharded_filename, 1);
  }

  {
    int tensor_size =
        has_ev ? tensor_names_vec.size() + 1 : tensor_names_vec.size();
    // tensor_names
    Tensor new_tensor_names, new_tensor_shape;
    TensorProto tensor_shape_proto, tensor_name_proto;
    tensor_name_proto.set_dtype(DT_STRING);
    tensor_shape_proto.set_dtype(DT_STRING);
    TensorShape({tensor_size})
        .AsProto(tensor_shape_proto.mutable_tensor_shape());
    TensorShape({tensor_size})
        .AsProto(tensor_name_proto.mutable_tensor_shape());
    if (has_ev) {
      tensor_name_proto.add_string_val("global_step");
      tensor_shape_proto.add_string_val("");
    }
    for (int j = 0; j < tensor_names_vec.size(); ++j) {
      tensor_name_proto.add_string_val(tensor_names_vec[j]);
    }
    bool ret = new_tensor_names.FromProto(tensor_name_proto);
    if (!ret) return errors::Internal("tensor_name tensor init error");
    NodeDef tensor_name_node_def;
    TF_RETURN_IF_ERROR(NodeDefBuilder(ori_save_node->name() + "_" +
                                          std::to_string(i) + "/tensor_names",
                                      "Const")
                           .Attr("value", new_tensor_names)
                           .Attr("dtype", DT_STRING)
                           .Device(assigned_device_name)
                           .Finalize(&tensor_name_node_def));
    tensor_name_node = g->AddNode(tensor_name_node_def, &s);
    tensor_name_node->set_assigned_device_name(assigned_device_name);
    TF_RETURN_IF_ERROR(s);

    std::vector<string> new_tensor_shape_vec;
    for (int j = 0; j < tensor_names_vec.size(); ++j) {
      string tensor_n = tensor_names_vec[j];
      auto it = variable_shape.find(tensor_n);
      if (it != variable_shape.end()) {
        string tmp_shape_and_slice = "";
        auto shape_and_slice = it->second;
        for (int j = 0; j < shape_and_slice.size() / 2; ++j) {
          tmp_shape_and_slice += std::to_string(shape_and_slice[j]);
          tmp_shape_and_slice += " ";
        }
        std::vector<string> tmp_dim;
        for (int j = 0; j < shape_and_slice.size() / 2; ++j) {
          // partition_idx
          if (j == 0) {
            int remainder = shape_and_slice[j] % cur_partition_nums;
            int64 low = 0;
            int64 tmp_part = i;
            while (tmp_part-- > 0) {
              if (remainder > tmp_part) {
                low += shape_and_slice[j + shape_and_slice.size() / 2] + 1;
              } else {
                low += shape_and_slice[j + shape_and_slice.size() / 2];
              }
            }
            int64 high = shape_and_slice[j + shape_and_slice.size() / 2];
            if (remainder > i) {
              high += 1;
            }
            tmp_dim.emplace_back(std::to_string(low) + "," +
                                 std::to_string(high));
          } else {
            int64 low = 0;
            int64 high = shape_and_slice[j];
            tmp_dim.emplace_back(std::to_string(low) + "," +
                                 std::to_string(high));
          }
        }
        tmp_shape_and_slice += str_util::Join(tmp_dim, ":");
        tensor_shape_proto.add_string_val(tmp_shape_and_slice);
      }
    }
    ret = new_tensor_shape.FromProto(tensor_shape_proto);
    if (!ret) return errors::Internal("tensor_name tensor init error");
    NodeDef shape_slice_node_def;
    TF_RETURN_IF_ERROR(NodeDefBuilder(ori_save_node->name() + "_" +
                                          std::to_string(i) +
                                          "/shape_and_slices",
                                      "Const")
                           .Attr("value", new_tensor_shape)
                           .Attr("dtype", DT_STRING)
                           .Device(assigned_device_name)
                           .Finalize(&shape_slice_node_def));
    shape_slice_node = g->AddNode(shape_slice_node_def, &s);
    shape_slice_node->set_assigned_device_name(assigned_device_name);
    TF_RETURN_IF_ERROR(s);
  }

  {
    // ev_names
    NodeDef ev_name_node_def;
    Tensor ev_names_tensor;
    TensorProto ev_names_proto;
    ev_names_proto.set_dtype(DT_STRING);
    TensorShape({static_cast<int64>(ev_names_vec.size())})
        .AsProto(ev_names_proto.mutable_tensor_shape());
    for (int k = 0; k < ev_names_vec.size(); ++k) {
      ev_names_proto.add_string_val(ev_names_vec[k]);
    }
    bool ret = ev_names_tensor.FromProto(ev_names_proto);
    if (!ret) return errors::Internal("tensor_name tensor init error");
    TF_RETURN_IF_ERROR(NodeDefBuilder(ori_save_node->name() + "_" +
                                          std::to_string(i) + "/ev_names",
                                      "Const")
                           .Attr("value", ev_names_tensor)
                           .Attr("dtype", DT_STRING)
                           .Device(assigned_device_name)
                           .Finalize(&ev_name_node_def));
    ev_name_node = g->AddNode(ev_name_node_def, &s);
    ev_name_node->set_assigned_device_name(assigned_device_name);
    TF_RETURN_IF_ERROR(s);
  }

  {
    std::vector<NodeDefBuilder::NodeOut> kv_lookup_resource_input;
    for (auto* n : kv_lookup_resource_node_vec) {
      kv_lookup_resource_input.emplace_back(n->name(), 0, n->output_type(0));
      ev_dtypes.emplace_back(DT_INT64);
    }
    DataType key_type;
    if (key_data_types.size() == 0) {
      key_type = DT_INT64;
      Tensor const_tensor(DT_INT64, TensorShape({}));
      // ev_resources
      NodeDef const_node_def;
      TF_RETURN_IF_ERROR(NodeDefBuilder(ori_save_node->name() + "_" +
                                            std::to_string(i) + "/ev_resources",
                                        "Const")
                             .Attr("dtype", key_type)
                             .Attr("value", const_tensor)
                             .Device(assigned_device_name)
                             .Finalize(&const_node_def));
      kv_lookup_resource_node = g->AddNode(const_node_def, &s);
      kv_lookup_resource_node->set_assigned_device_name(assigned_device_name);
      TF_RETURN_IF_ERROR(s);
    } else {
      key_type = key_data_types[0];
      // ev_resources
      NodeDef kv_lookup_resource_node_def;
      int n = kv_lookup_resource_node_vec.size();
      TF_RETURN_IF_ERROR(NodeDefBuilder(ori_save_node->name() + "_" +
                                            std::to_string(i) + "/ev_resources",
                                        "Pack")
                             .Input(kv_lookup_resource_input)
                             .Attr("N", n)
                             .Attr("T", key_type)
                             .Attr("axis", 0)
                             .Device(assigned_device_name)
                             .Finalize(&kv_lookup_resource_node_def));
      kv_lookup_resource_node = g->AddNode(kv_lookup_resource_node_def, &s);
      kv_lookup_resource_node->set_assigned_device_name(assigned_device_name);
      TF_RETURN_IF_ERROR(s);
    }
  }
  if (n_dtypes.size() > 0) {
    // tensor_names
    NodeDef save_node_def;
    TF_RETURN_IF_ERROR(
        NodeDefBuilder(ori_save_node->name() + "_" + std::to_string(i), kSaveOp)
            .Input((*new_sharded_filename)->name(), 0,
                   (*new_sharded_filename)->output_type(0))
            .Input(tensor_name_node->name(), 0,
                   tensor_name_node->output_type(0))
            .Input(shape_slice_node->name(), 0,
                   shape_slice_node->output_type(0))
            .Input(ev_name_node->name(), 0, ev_name_node->output_type(0))
            .Input(kv_lookup_resource_node->name(), 0,
                   kv_lookup_resource_node->output_type(0))
            .Input(tensors_input)
            .Attr("ev_key_types", ev_dtypes)
            .Attr("has_ev", has_ev)
            .Attr("dtypes", n_dtypes)
            .Device(assigned_device_name)
            .Finalize(&save_node_def));
    Node* save_node = g->AddNode(save_node_def, &s);
    TF_RETURN_IF_ERROR(s);
    save_node->set_assigned_device_name(assigned_device_name);

    for (auto* o_edge : ori_save_node->out_edges()) {
      if (o_edge->IsControlEdge()) {
        Node* save_control_node =
            CopyNode(g, o_edge->dst(), assigned_device_name, i);
        g->AddEdge(*new_sharded_filename, 0, save_control_node, 0);
        g->AddControlEdge(save_node, save_control_node);
        for (auto* oo_edge : o_edge->dst()->out_edges()) {
          if (oo_edge->IsControlEdge()) {
            auto* dst_node = oo_edge->dst();
            g->AddControlEdge(save_control_node, dst_node);
            if (dst_node->type_string() == "Pack") {
              int part_num;
              TF_RETURN_IF_ERROR(
                  GetNodeAttr(dst_node->attrs(), "N", &part_num));
              if (part_num != cur_partition_nums) {
                dst_node->ClearAttr("N");
                dst_node->AddAttr("N", cur_partition_nums);
              }
              g->AddEdge(*new_sharded_filename, 0, dst_node, i);
            }
          }
        }
      }
    }
  }
  return s;
}

Status AddNewRestoreGraph(
    Graph* g, bool has_ev, int i, int cur_partition_nums,
    const Node* new_sharded_filename,
    const std::vector<string>& tensor_names_vec,
    const std::unordered_map<string, std::vector<int64>>& variable_shape,
    const std::vector<Node*>& restore_tensor_vec,
    const std::vector<Node*>& restore_node_vec,
    const std::string& assigned_device_name, std::vector<DataType>& n_dtypes) {
  Status s;
  Node* ori_restore_node = restore_node_vec[0];
  Node* restore_tensor_name_node;
  Node* restore_shape_slice_node;
  {
    int tensor_size = tensor_names_vec.size();
    // tensor_names
    Tensor new_tensor_names, new_tensor_shape;
    TensorProto tensor_shape_proto, tensor_name_proto;
    tensor_name_proto.set_dtype(DT_STRING);
    tensor_shape_proto.set_dtype(DT_STRING);
    TensorShape({tensor_size})
        .AsProto(tensor_shape_proto.mutable_tensor_shape());
    TensorShape({tensor_size})
        .AsProto(tensor_name_proto.mutable_tensor_shape());
    for (int j = 0; j < tensor_names_vec.size(); ++j) {
      tensor_name_proto.add_string_val(tensor_names_vec[j]);
    }
    bool ret = new_tensor_names.FromProto(tensor_name_proto);
    if (!ret) return errors::Internal("tensor_name tensor init error");
    NodeDef tensor_name_node_def;
    TF_RETURN_IF_ERROR(NodeDefBuilder(ori_restore_node->name() + "_" +
                                          std::to_string(i) + "/tensor_names",
                                      "Const")
                           .Attr("value", new_tensor_names)
                           .Attr("dtype", DT_STRING)
                           .Device(assigned_device_name)
                           .Finalize(&tensor_name_node_def));
    restore_tensor_name_node = g->AddNode(tensor_name_node_def, &s);
    TF_RETURN_IF_ERROR(s);
    restore_tensor_name_node->set_assigned_device_name(assigned_device_name);
    std::vector<string> new_tensor_shape_vec;
    for (int j = 0; j < tensor_names_vec.size(); ++j) {
      string tensor_n = tensor_names_vec[j];
      auto it = variable_shape.find(tensor_n);
      if (it != variable_shape.end()) {
        string tmp_shape_and_slice = "";
        auto shape_and_slice = it->second;
        for (int j = 0; j < shape_and_slice.size() / 2; ++j) {
          tmp_shape_and_slice += std::to_string(shape_and_slice[j]);
          tmp_shape_and_slice += " ";
        }
        std::vector<string> tmp_dim;
        for (int j = 0; j < shape_and_slice.size() / 2; ++j) {
          // partition_idx
          if (j == 0) {
            int64 low = i * shape_and_slice[j + shape_and_slice.size() / 2];
            int64 high;
            if (i == cur_partition_nums - 1) {
              high = shape_and_slice[j] - low;
            } else {
              high = shape_and_slice[j + shape_and_slice.size() / 2];
            }

            tmp_dim.emplace_back(std::to_string(low) + "," +
                                 std::to_string(high));
          } else {
            int64 low = 0;
            int64 high = shape_and_slice[j];
            tmp_dim.emplace_back(std::to_string(low) + "," +
                                 std::to_string(high));
          }
        }
        tmp_shape_and_slice += str_util::Join(tmp_dim, ":");
        tensor_shape_proto.add_string_val(tmp_shape_and_slice);
      }
    }
    ret = new_tensor_shape.FromProto(tensor_shape_proto);
    if (!ret) return errors::Internal("tensor_name tensor init error");
    NodeDef shape_slice_node_def;
    TF_RETURN_IF_ERROR(NodeDefBuilder(ori_restore_node->name() + "_" +
                                          std::to_string(i) +
                                          "/shape_and_slices",
                                      "Const")
                           .Attr("value", new_tensor_shape)
                           .Attr("dtype", DT_STRING)
                           .Device(assigned_device_name)
                           .Finalize(&shape_slice_node_def));
    restore_shape_slice_node = g->AddNode(shape_slice_node_def, &s);
    TF_RETURN_IF_ERROR(s);
    restore_shape_slice_node->set_assigned_device_name(assigned_device_name);
  }

  NodeDef restore_node_def;
  if (has_ev) {
    n_dtypes.erase(n_dtypes.begin());
  }
  if (n_dtypes.size() > 0) {
    TF_RETURN_IF_ERROR(
        NodeDefBuilder(ori_restore_node->name() + "_" + std::to_string(i),
                       kRestoreOp)
            .Input(new_sharded_filename->name(), 0,
                   new_sharded_filename->output_type(0))
            .Input(restore_tensor_name_node->name(), 0,
                   restore_tensor_name_node->output_type(0))
            .Input(restore_shape_slice_node->name(), 0,
                   restore_shape_slice_node->output_type(0))
            .Attr("dtypes", n_dtypes)
            .Device(assigned_device_name)
            .Finalize(&restore_node_def));
    Node* restore_node = g->AddNode(restore_node_def, &s);
    TF_RETURN_IF_ERROR(s);
    restore_node->set_assigned_device_name(assigned_device_name);
    NodeDef restore_no_op_def;
    TF_RETURN_IF_ERROR(
        NodeDefBuilder("save/restore_all/NoOp_" + std::to_string(i), "NoOp")
            .Device(assigned_device_name)
            .Finalize(&restore_no_op_def));
    Node* restore_no_op = g->AddNode(restore_no_op_def, &s);
    TF_RETURN_IF_ERROR(s);
    restore_no_op->set_assigned_device_name(assigned_device_name);

    for (int k = 0; k < restore_tensor_vec.size(); ++k) {
      for (auto* o_node : restore_tensor_vec[k]->out_nodes()) {
        if (o_node->type_string() == "Assign") {
          Node* restore_assign_node = CopyNode(g, o_node, assigned_device_name,
                                               i, o_node->name() + "/Copy");
          g->AddEdge(restore_tensor_vec[k], 0, restore_assign_node, 0);
          g->AddEdge(restore_node, k, restore_assign_node, 1);
          g->AddControlEdge(restore_assign_node, restore_no_op);
        }
      }
    }
    // TODO(JUNQI): Dangerous!! Unexpected behaviour would be happened when
    // there is
    //              multiple SaverSubGraph.
    for (auto* n : g->nodes()) {
      if (n->name() == "save/restore_all") {
        g->AddControlEdge(restore_no_op, n);
        break;
      }
    }
  }
  return s;
}

Status RewritePrevSubGraph(
    Graph* g, int i, int cur_partition_nums, bool scaling_up,
    const std::vector<Node*>& save_node_vec,
    std::unordered_map<std::string, PartitionedVariable>&
        primary_node_metas_map,
    std::unordered_map<std::string, Node*>& unpartitioned_node_map,
    std::unordered_map<string, std::vector<int64>>& variable_shape,
    const std::unordered_map<Node*, std::pair<string, string>>& nodes_to_add,
    std::unordered_set<Node*>& nodes_to_delete,
    std::unordered_set<Node*>& eval_nodes_to_add,
    std::unordered_map<string, string>& opt_to_primary_map) {
  Status s;
  Node* ori_save_node = save_node_vec[i];
  string assigned_device_name = ori_save_node->assigned_device_name();

  Node* tensor_names;
  TF_RETURN_IF_ERROR(ori_save_node->input_node(1, &tensor_names));
  nodes_to_delete.insert(tensor_names);
  Node* shape_and_slices;
  TF_RETURN_IF_ERROR(ori_save_node->input_node(2, &shape_and_slices));
  nodes_to_delete.insert(shape_and_slices);
  Tensor tensor_name_t;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(tensor_names->attrs(), "value", &tensor_name_t));
  Tensor shape_and_slice_t;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(shape_and_slices->attrs(), "value", &shape_and_slice_t));
  std::vector<string> new_tensor_shape;
  std::vector<string> new_tensor_name;
  for (int k = 0; k < tensor_name_t.dim_size(0); ++k) {
    string tensor_n = tensor_name_t.flat<tstring>()(k);
    auto s_and_s_s = shape_and_slice_t.flat<tstring>()(k);
    new_tensor_name.emplace_back(tensor_n);
    Node* input_node = nullptr;
    TF_RETURN_IF_ERROR(ori_save_node->input_node(5 + k, &input_node));
    Node* actual_node = nullptr;
    int actual_part_id = 0;
    if (input_node->type_string() == "Identity") {
      Node* identity_node;
      TF_RETURN_IF_ERROR(input_node->input_node(0, &identity_node));
      if (identity_node->type_string() == "Identity") {
        const Node* read_variable_node;
        TF_RETURN_IF_ERROR(identity_node->input_node(0, &read_variable_node));
        Node* variable_node;
        TF_RETURN_IF_ERROR(read_variable_node->input_node(0, &variable_node));
        auto part_it = unpartitioned_node_map.find(variable_node->name());
        if (part_it != unpartitioned_node_map.end()) {
          eval_nodes_to_add.emplace(variable_node);
        } else {
          actual_node = variable_node;
        }
      } else {  // RefVariable
        if (identity_node->name() != "global_step") {
          auto part_it = unpartitioned_node_map.find(identity_node->name());
          if (part_it != unpartitioned_node_map.end()) {
            eval_nodes_to_add.emplace(identity_node);
          } else {
            actual_node = identity_node;
            int part_id = GetNodePartIndex(actual_node);
            if ((part_id >= cur_partition_nums) && !scaling_up) {
              auto tmp_it = primary_node_metas_map.find(tensor_n);
              if (tmp_it != primary_node_metas_map.end()) {
                actual_part_id = tmp_it->second.node_device_map[actual_node];
              } else if (opt_to_primary_map.find(tensor_n) !=
                         opt_to_primary_map.end()) {
                auto t_it =
                    primary_node_metas_map.find(opt_to_primary_map[tensor_n]);
                actual_part_id = t_it->second.node_device_map[actual_node];
              }
            }
          }
        }
      }
    } else if (input_node->type_string() == "ReadVariableOp") {
      Node* read_variable_node;
      TF_RETURN_IF_ERROR(input_node->input_node(0, &read_variable_node));
      auto part_it = unpartitioned_node_map.find(read_variable_node->name());
      if (part_it != unpartitioned_node_map.end()) {
        eval_nodes_to_add.emplace(read_variable_node);
      } else {
        actual_node = read_variable_node;
      }
    } else {  // moving_average
      auto part_it = unpartitioned_node_map.find(input_node->name());
      if (part_it != unpartitioned_node_map.end()) {
        eval_nodes_to_add.emplace(input_node);
      } else {
        actual_node = input_node;
      }
    }
    auto it = variable_shape.find(tensor_n);
    if (it ==
        variable_shape.end()) {  // TODO (JUNQI): which variable is this case
      new_tensor_shape.emplace_back(s_and_s_s);
    } else {
      int part_id = GetNodePartIndex(actual_node);
      if (part_id >= cur_partition_nums) {
        part_id = actual_part_id;
      }
      auto s_and_s_s = shape_and_slice_t.flat<tstring>()(k);
      std::vector<string> splits = str_util::Split(s_and_s_s, ' ');
      if (splits.size() < 2) {
        LOG(ERROR)
            << "Need least two elements in shape_and_slice specification: ";
      }
      std::vector<string> items =
          str_util::Split(splits.back(), ':', str_util::SkipEmpty());
      std::vector<int64> shape_and_slice(items.size() * 2, 1);
      int remainder = 0;
      for (int j = 0; j < items.size(); ++j) {
        int64 dim;
        if (!strings::safe_strto64(splits[j], &dim)) {
          LOG(ERROR) << "Non numerical dimension in shape_and_slice: ";
        }
        // partition_idx
        if (j == 0) {
          shape_and_slice[j] = dim;
          // TODO(JUNQI) : partition_num 1 is impossible
          shape_and_slice[j + items.size()] = dim / cur_partition_nums;
          remainder = dim % cur_partition_nums;
        } else {
          shape_and_slice[j] = dim;
          shape_and_slice[j + items.size()] = dim;
        }
      }

      string tmp_shape_and_slice = "";
      for (int j = 0; j < shape_and_slice.size() / 2; ++j) {
        tmp_shape_and_slice += std::to_string(shape_and_slice[j]);
        tmp_shape_and_slice += " ";
      }
      std::vector<string> tmp_dim;
      for (int j = 0; j < shape_and_slice.size() / 2; ++j) {
        // partition_idx
        if (j == 0) {
          int64 low = 0;
          int tmp_part = part_id;
          while (tmp_part-- > 0) {
            if (remainder > tmp_part) {
              low += shape_and_slice[j + shape_and_slice.size() / 2] + 1;
            } else {
              low += shape_and_slice[j + shape_and_slice.size() / 2];
            }
          }
          if (remainder > part_id) {
            int64 high = shape_and_slice[j + shape_and_slice.size() / 2] + 1;
            tmp_dim.emplace_back(std::to_string(low) + "," +
                                 std::to_string(high));
          } else {
            int64 high = shape_and_slice[j + shape_and_slice.size() / 2];
            tmp_dim.emplace_back(std::to_string(low) + "," +
                                 std::to_string(high));
          }
        } else {
          int64 low = 0;
          int64 high = shape_and_slice[j];
          tmp_dim.emplace_back(std::to_string(low) + "," +
                               std::to_string(high));
        }
      }
      tmp_shape_and_slice += str_util::Join(tmp_dim, ":");
      new_tensor_shape.emplace_back(tmp_shape_and_slice);
      variable_shape[tensor_n] = shape_and_slice;
    }
  }
  int old_tensor_size = new_tensor_shape.size();

  // place nodes on ps which is scaled out on ps:0
  if ((i == 0) && (nodes_to_add.size() > 0)) {
    std::vector<DataType> n_dtypes;
    TF_RETURN_IF_ERROR(
        GetNodeAttr(ori_save_node->attrs(), "dtypes", &n_dtypes));
    int k = 0;
    for (auto& it : nodes_to_add) {
      new_tensor_name.emplace_back(it.second.first);
      new_tensor_shape.emplace_back(it.second.second);
      if (it.second.first == "global_step") {
        n_dtypes.emplace_back(DT_INT64);
      } else {
        n_dtypes.emplace_back(DT_FLOAT);
      }
      g->AddEdge(it.first, 0, ori_save_node, 5 + old_tensor_size + k);
      ++k;
    }
    ori_save_node->ClearAttr("dtypes");
    ori_save_node->AddAttr("dtypes", n_dtypes);
  }

  int tensor_size = new_tensor_shape.size();
  // tensor_names
  Tensor new_tensor_name_t;
  TensorProto tensor_name_proto;
  tensor_name_proto.set_dtype(DT_STRING);
  TensorShape({tensor_size}).AsProto(tensor_name_proto.mutable_tensor_shape());
  for (int j = 0; j < new_tensor_name.size(); ++j) {
    tensor_name_proto.add_string_val(new_tensor_name[j]);
  }
  bool ret = new_tensor_name_t.FromProto(tensor_name_proto);
  if (!ret) return errors::Internal("tensor_name tensor init error");
  NodeDef name_node_def;
  TF_RETURN_IF_ERROR(NodeDefBuilder(tensor_names->name() + "/Copy", "Const")
                         .Attr("value", new_tensor_name_t)
                         .Attr("dtype", DT_STRING)
                         .Device(assigned_device_name)
                         .Finalize(&name_node_def));
  Node* name_node = g->AddNode(name_node_def, &s);
  TF_RETURN_IF_ERROR(s);
  name_node->set_assigned_device_name(assigned_device_name);
  TF_RETURN_IF_ERROR(g->UpdateEdge(name_node, 0, ori_save_node, 1));

  Tensor new_tensor_shape_t;
  TensorProto tensor_shape_proto;
  tensor_shape_proto.set_dtype(DT_STRING);
  TensorShape({tensor_size}).AsProto(tensor_shape_proto.mutable_tensor_shape());
  for (int j = 0; j < new_tensor_shape.size(); ++j) {
    tensor_shape_proto.add_string_val(new_tensor_shape[j]);
  }
  ret = new_tensor_shape_t.FromProto(tensor_shape_proto);
  if (!ret) return errors::Internal("shape tensor init error");
  NodeDef shape_slice_node_def;
  TF_RETURN_IF_ERROR(NodeDefBuilder(shape_and_slices->name() + "/Copy", "Const")
                         .Attr("value", new_tensor_shape_t)
                         .Attr("dtype", DT_STRING)
                         .Device(assigned_device_name)
                         .Finalize(&shape_slice_node_def));
  Node* shape_slice_node = g->AddNode(shape_slice_node_def, &s);
  TF_RETURN_IF_ERROR(s);
  shape_slice_node->set_assigned_device_name(assigned_device_name);
  TF_RETURN_IF_ERROR(g->UpdateEdge(shape_slice_node, 0, ori_save_node, 2));

  return s;
}

Status DeleteOldSaverGraph(
    const std::vector<Node*>& save_node_vec,
    const std::unordered_map<string, std::vector<int64>>
        partitioned_variable_shape,
    const std::unordered_map<std::string, Node*>& unpartitioned_node_map, int i,
    int cur_partition_nums,
    std::unordered_map<Node*, std::pair<string, string>>& nodes_to_add,
    std::unordered_set<Node*>& nodes_to_delete) {
  Node* cur_save_node = save_node_vec[i];
  {
    Node* sharded_filename;
    TF_RETURN_IF_ERROR(cur_save_node->input_node(0, &sharded_filename));
    for (auto* o_node : sharded_filename->out_nodes()) {
      if (o_node->type_string() == "Identity") {
        nodes_to_delete.insert(o_node);
      } else if (o_node->type_string() == "Pack") {
        int part_num;
        TF_RETURN_IF_ERROR(GetNodeAttr(o_node->attrs(), "N", &part_num));
        if (part_num != cur_partition_nums) {
          o_node->ClearAttr("N");
          o_node->AddAttr("N", cur_partition_nums);
        }
      }
    }
    nodes_to_delete.insert(sharded_filename);
  }

  {
    Node* tensor_names;
    TF_RETURN_IF_ERROR(cur_save_node->input_node(1, &tensor_names));
    Tensor tensor_name_t;
    TF_RETURN_IF_ERROR(
        GetNodeAttr(tensor_names->attrs(), "value", &tensor_name_t));
    Node* shape_and_slices;
    TF_RETURN_IF_ERROR(cur_save_node->input_node(2, &shape_and_slices));
    Tensor shape_and_slice_t;
    TF_RETURN_IF_ERROR(
        GetNodeAttr(shape_and_slices->attrs(), "value", &shape_and_slice_t));
    for (int k = 0; k < tensor_name_t.dim_size(0); ++k) {
      string tensor_n = tensor_name_t.flat<tstring>()(k);
      string s_and_s_s = shape_and_slice_t.flat<tstring>()(k);
      if (partitioned_variable_shape.find(tensor_n) ==
          partitioned_variable_shape.end()) {
        Node* n;
        TF_RETURN_IF_ERROR(cur_save_node->input_node(5 + k, &n));
        if (n->name() == "global_step/read") {
          continue;
        }
        if (n->name() == tensor_n) {
          nodes_to_add.emplace(n,
                               std::pair<string, string>(tensor_n, s_and_s_s));
        } else if (n->type_string() == "Identity") {
          Node* identity_node;
          TF_RETURN_IF_ERROR(n->input_node(0, &identity_node));
          if (identity_node->type_string() == "Identity") {  // ResourceVariable
            Node* read_variable_node;
            TF_RETURN_IF_ERROR(
                identity_node->input_node(0, &read_variable_node));
            Node* resource_node;
            TF_RETURN_IF_ERROR(
                read_variable_node->input_node(0, &resource_node));
            if (unpartitioned_node_map.find(resource_node->name()) !=
                unpartitioned_node_map.end()) {
              nodes_to_add.emplace(
                  n, std::pair<string, string>(tensor_n, s_and_s_s));
            }
          } else {  // RefVariable
            nodes_to_add.emplace(
                n, std::pair<string, string>(tensor_n, s_and_s_s));
          }
        }
      }
    }
    nodes_to_delete.insert(tensor_names);
    nodes_to_delete.insert(shape_and_slices);
  }

  {
    Node* ev_names;
    TF_RETURN_IF_ERROR(cur_save_node->input_node(3, &ev_names));
    nodes_to_delete.insert(ev_names);
    Node* ev_resource;
    TF_RETURN_IF_ERROR(cur_save_node->input_node(4, &ev_resource));
    nodes_to_delete.insert(ev_resource);
  }
  nodes_to_delete.insert(cur_save_node);
  return Status::OK();
}

void MovePartitionedVariable(
    Graph* g, int cur_partition_nums,
    std::unordered_map<std::string, PartIdToNodeMap>& node_to_origin_map,
    const std::string& primary_variable_name,
    const std::vector<std::string>& opt_var_names,
    const std::vector<int>& part_to_move, ElasticHookMetaNode* meta_node) {
  int assigned_device_index = 0;
  auto func = [&assigned_device_index, cur_partition_nums](
                  Graph* g, Node* init_op, Node* var_node) {
    var_node->set_assigned_device_name(init_op->assigned_device_name());
    for (auto* o_node : var_node->out_nodes()) {
      o_node->set_assigned_device_name(var_node->assigned_device_name());
      if (o_node->type_string() == ::des::kReAssign) {
        o_node->ClearAttr("partition_nums");
        o_node->AddAttr("partition_nums", cur_partition_nums + 1);
        o_node->ClearAttr("device_id");
        o_node->AddAttr("device_id", assigned_device_index);
        o_node->ClearAttr("partition_id");
        o_node->AddAttr("partition_id", cur_partition_nums);
        o_node->set_assigned_device_name(init_op->assigned_device_name());
        g->AddControlEdge(o_node, init_op);
        Node* value_node = nullptr;
        Status s = o_node->input_node(1, &value_node);
        if (!s.ok()) LOG(ERROR) << s.error_message();
        for (auto* i_edge : value_node->in_edges()) {
          if (i_edge->IsControlEdge()) {
            g->RemoveControlEdge(i_edge);
          }
        }
      }
    }
  };

  for (auto& i : part_to_move) {
    auto* init_op =
        meta_node->m_init_op_vec[assigned_device_index % cur_partition_nums];
    auto* var_node = node_to_origin_map[primary_variable_name][i];
    func(g, init_op, var_node);
    for (auto& opt_name : opt_var_names) {
      auto* opt_var_node = node_to_origin_map[opt_name][i];
      func(g, init_op, opt_var_node);
    }
    ++assigned_device_index;
  }
}

Status MoveUnPartitionedVariable(Graph* g, Node* search_node,
                                 ElasticHookMetaNode& meta_node) {
  Node* target_node = nullptr;
  if (search_node->type_string() == "VariableV2") {
    target_node = search_node;
  } else if (search_node->type_string() == "VarHandleOp") {
    target_node = search_node;
  } else if (search_node->IsKvVarHandle()) {
    return Status::OK();
  } else {
    Node* identity_node;
    TF_RETURN_IF_ERROR(search_node->input_node(0, &identity_node));
    if (identity_node->type_string() == "Identity") {
      Node* read_variable_node;
      TF_RETURN_IF_ERROR(identity_node->input_node(0, &read_variable_node));
      Node* resource_node;
      TF_RETURN_IF_ERROR(read_variable_node->input_node(0, &resource_node));
      target_node = resource_node;
    } else {
      target_node = identity_node;
    }
  }

  for (auto* o_node : target_node->out_nodes()) {
    if (o_node->name().find("elastic_import") != string::npos) {
      std::vector<Node*> node_to_del;
      string assigned_device_name;
      Node* read_value_node = nullptr;
      if (o_node->type_string() == "Identity") {
        node_to_del.emplace_back(o_node);
        for (auto* oo_node : o_node->out_nodes()) {
          if (oo_node->type_string() == "AssignVariableOp") {
            node_to_del.emplace_back(oo_node);
            Node* tmp_value;
            TF_RETURN_IF_ERROR(oo_node->input_node(0, &tmp_value));
            for (auto* ooo_node : tmp_value->out_nodes()) {
              if (ooo_node->type_string() == "ReadVariableOp") {
                read_value_node = ooo_node;
              }
            }
            assigned_device_name = oo_node->assigned_device_name();
            meta_node.m_tmp_value_init_op->set_assigned_device_name(
                assigned_device_name);
            target_node->set_assigned_device_name(assigned_device_name);
          }
        }

      } else if (o_node->type_string() == "ReadVariableOp") {
        for (auto* oo_node : o_node->out_nodes()) {
          if (oo_node->type_string() == "Identity") {
            for (auto* ooo_node : oo_node->out_nodes()) {
              if (ooo_node->type_string() == "AssignVariableOp") {
                Node* tmp_value;
                TF_RETURN_IF_ERROR(ooo_node->input_node(0, &tmp_value));
                o_node->set_assigned_device_name(
                    tmp_value->assigned_device_name());
                meta_node.m_tmp_value_init_op->set_assigned_device_name(
                    ooo_node->assigned_device_name());
                target_node->set_assigned_device_name(
                    tmp_value->assigned_device_name());
                TF_RETURN_IF_ERROR(g->UpdateEdge(tmp_value, 0, o_node, 0));
                TF_RETURN_IF_ERROR(g->UpdateEdge(target_node, 0, ooo_node, 0));
                TF_RETURN_IF_ERROR(g->UpdateEdge(oo_node, 0, ooo_node, 1));
                g->AddControlEdge(ooo_node, meta_node.m_tmp_value_init_op);
                for (auto* tmp_out : target_node->out_nodes()) {
                  tmp_out->set_assigned_device_name(
                      tmp_value->assigned_device_name());
                }
                return Status::OK();
              }
            }
          }
        }
      }
      for (auto* node : node_to_del) {
        g->RemoveNode(node);
      }

      if (read_value_node != nullptr) {
        Status s;
        NodeDef assignop_def;
        TF_RETURN_IF_ERROR(
            NodeDefBuilder("elastic_import/ASSIGN/" + target_node->name(),
                           "Assign")
                .Input(target_node->name(), 0, target_node->output_type(0))
                .Input(read_value_node->name(), 0,
                       read_value_node->output_type(0))
                .Attr("T", read_value_node->output_type(0))
                .Attr("validate_shape", false)
                .Attr("use_locking", true)
                .Device(assigned_device_name)
                .Finalize(&assignop_def));
        Node* assign_node = g->AddNode(assignop_def, &s);
        TF_RETURN_IF_ERROR(s);
        g->AddEdge(target_node, 0, assign_node, 0);
        g->AddEdge(read_value_node, 0, assign_node, 1);
        g->AddControlEdge(assign_node, meta_node.m_tmp_value_init_op);
      }
    }
  }
  return Status::OK();
}

Status DeleteUnlessUnPartitionedVariable(Graph* g, Node* target_node) {
  for (auto* o_node : target_node->out_nodes()) {
    if ((o_node->type_string() == "Identity") &&
        (o_node->name().find("elastic_import") != string::npos)) {
      for (auto* oo_node : o_node->out_nodes()) {
        if (oo_node->type_string() == "Assign") {
          g->RemoveNode(oo_node);
        }
      }
    } else if ((o_node->type_string() == "ReadVariableOp") &&
               (o_node->name().find("elastic_import") != string::npos)) {
      // ReadVariable -> Identity -> Identity
      for (auto* oo_node : o_node->out_nodes()) {
        if (oo_node->type_string() == "Identity") {
          for (auto* ooo_node : oo_node->out_nodes()) {
            if ((ooo_node->type_string() == "AssignVariableOp") &&
                (ooo_node->name().find("elastic_import") != string::npos)) {
              g->RemoveNode(ooo_node);
            }
          }
        }
      }
    }
  }
  return Status::OK();
}

Status ProcessUnPartitionedVariable(
    Graph* g, ElasticHookMetaNode& meta_node,
    std::unordered_map<Node*, std::pair<string, string>>& nodes_to_add,
    std::unordered_set<Node*>& eval_nodes_to_add,
    std::unordered_map<std::string, Node*>& unpartitioned_node_map) {
  for (auto& it : unpartitioned_node_map) {
    if (eval_nodes_to_add.find(it.second) != eval_nodes_to_add.end()) {
      TF_RETURN_IF_ERROR(DeleteUnlessUnPartitionedVariable(g, it.second));
    } else if (it.first == kWorkerSyncOp) {
      continue;
    } else {
      TF_RETURN_IF_ERROR(MoveUnPartitionedVariable(g, it.second, meta_node));
    }
  }
  return Status::OK();
}

}  // namespace tensorflow

#endif  // DYNMAMIC_EMBEDDING_SERVER_INCLUDE_UTILS_GRAPH_UTILS_H_
