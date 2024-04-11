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

constexpr char kPart[] = "part_";
constexpr char kDynamicPartition[] = "DynamicPartition";
constexpr char kParaDynamicStitch[] = "ParallelDynamicStitch";
constexpr char kIdentityOp[] = "Identity";
constexpr char kVariableOp[] = "VariableV2";
constexpr char kRestoreOp[] = "RestoreV2";
constexpr char kWorkerSyncOp[] = "worker_sync";

namespace tensorflow {

typedef std::pair<int, Node*> DeviceIdToNode;

constexpr char kEvInitOp[] = "InitializeKvVariableV2Op";
constexpr char kSaveOp[] = "SaveV3";
constexpr int kSaverTensorInputOffset = 5;

bool IsUnique(Node* node) { return node->IsUnique(); }

void VLogGraphDebugString(Graph* g) {
  GraphDef graph_def;
  g->ToGraphDef(&graph_def);
  VLOG(1) << "Graph: " << graph_def.DebugString();
}

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

Status MakeConstTensor(Graph* g, Node* node, int partition_num, Node** new_node,
                       const string& new_device_name) {
  Status s;
  Tensor init_tensor(DT_INT32, {});
  init_tensor.flat<int>()(0) = partition_num;
  NodeDef init_value_def;
  TF_RETURN_IF_ERROR(
      NodeDefBuilder("Const/placeholder/" + node->name(), "Const")
          .Attr("dtype", DT_INT32)
          .Attr("value", init_tensor)
          .Device(new_device_name)
          .Finalize(&init_value_def));
  *new_node = g->AddNode(init_value_def, &s);
  TF_RETURN_IF_ERROR(s);
  (*new_node)->set_assigned_device_name(new_device_name);
  return s;
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

Status MakeTruncatedNormalInitializer(Graph* g, const Node* old_default_value,
                                      Node* new_gather_op, int i) {
  std::string prefix = new_gather_op->name();
  std::string new_device_name = new_gather_op->assigned_device_name();

  Node* default_v = CopyNode(g, old_default_value, new_device_name, i,
                             prefix + "/Initializer/truncated_normal");
  g->AddEdge(default_v, 0, new_gather_op, 2);

  // TrunMean
  Node* old_trun_mean = nullptr;
  TF_RETURN_IF_ERROR(old_default_value->input_node(1, &old_trun_mean));
  Node* trun_mean = CopyNode(g, old_trun_mean, new_device_name, i,
                             prefix + "/Initializer/truncated_normal/mean");
  g->AddEdge(trun_mean, 0, default_v, 1);

  // Mul
  Node* old_trun_mul = nullptr;
  TF_RETURN_IF_ERROR(old_default_value->input_node(0, &old_trun_mul));
  Node* trun_mul = CopyNode(g, old_trun_mul, new_device_name, i,
                            prefix + "/Initializer/truncated_normal/mul");
  g->AddEdge(trun_mul, 0, default_v, 0);

  // TrunNormal
  Node* old_truncated_normal = nullptr;
  TF_RETURN_IF_ERROR(old_trun_mul->input_node(0, &old_truncated_normal));
  Node* truncated_normal =
      CopyNode(g, old_truncated_normal, new_device_name, i,
               prefix + "/Initializer/truncated_normal/TruncatedNormal");
  g->AddEdge(truncated_normal, 0, trun_mul, 0);

  // Trunshape
  Node* old_trun_shape = nullptr;
  TF_RETURN_IF_ERROR(old_truncated_normal->input_node(0, &old_trun_shape));
  Node* trun_shape = CopyNode(g, old_trun_shape, new_device_name, i,
                              prefix + "/Initializer/truncated_normal/shape");
  g->AddEdge(trun_shape, 0, truncated_normal, 0);

  // TrunStdDev
  Node* old_trun_stddev = nullptr;
  TF_RETURN_IF_ERROR(old_trun_mul->input_node(1, &old_trun_stddev));
  Node* trun_stddev = CopyNode(g, old_trun_stddev, new_device_name, i,
                               prefix + "/Initializer/truncated_normal/stddev");
  g->AddEdge(trun_stddev, 0, trun_mul, 1);

  return Status::OK();
}

Status MakeEVInitializer(Node* ev_node, Node* new_ev_node, bool& is_init, int i,
                         Graph* g, Node* cur_init_op, Node** primary_init_node,
                         const std::string& new_device_name,
                         Node* new_opt_ev_node = nullptr) {
  // InitializeEVResource
  if (new_opt_ev_node == nullptr) {
    TF_RETURN_IF_ERROR(FindNodeAndExec(
        ev_node, kEvInitOp,
        [&primary_init_node, &new_ev_node, &is_init, &cur_init_op,
         new_device_name, g, i](Node* target_node) {
          if (!is_init) {
            const Node* tmp_check_ev_0;
            TF_RETURN_IF_ERROR(target_node->input_node(0, &tmp_check_ev_0));
            const Node* tmp_check_ev_1;
            TF_RETURN_IF_ERROR(target_node->input_node(1, &tmp_check_ev_1));
            // ensure initop is setted for primary ev
            if (tmp_check_ev_0->name() == tmp_check_ev_1->name()) {
              is_init = true;
              *primary_init_node = CopyNode(g, target_node, new_device_name, i);
              g->AddEdge(new_ev_node, 0, *primary_init_node, 0);
              g->AddEdge(new_ev_node, 0, *primary_init_node, 1);
              // init_value
              const Edge* init_value_edge = nullptr;
              TF_RETURN_IF_ERROR(target_node->input_edge(2, &init_value_edge));
              auto* init_value_node =
                  CopyNode(g, init_value_edge->src(), new_device_name, i);
              g->AddEdge(init_value_node, init_value_edge->src_output(),
                         *primary_init_node, 2);

              // empty_key
              const Edge* empty_key_edge = nullptr;
              TF_RETURN_IF_ERROR(target_node->input_edge(3, &empty_key_edge));
              auto* empty_key_node =
                  CopyNode(g, empty_key_edge->src(), new_device_name, i);
              g->AddEdge(empty_key_node, empty_key_edge->src_output(),
                         *primary_init_node, 3);
              g->AddControlEdge(*primary_init_node, cur_init_op);
            }
          }
          return Status::OK();
        }));
  } else {
    TF_RETURN_IF_ERROR(FindNodeAndExec(
        ev_node, kEvInitOp,
        [&primary_init_node, &new_ev_node, &cur_init_op, &is_init,
         &new_opt_ev_node, g, new_device_name, i](Node* target_node) {
          if (!is_init) {
            is_init = true;
            Node* init_node = CopyNode(g, target_node, new_device_name, i);
            g->AddEdge(new_opt_ev_node, 0, init_node, 0);
            g->AddEdge(new_ev_node, 0, init_node, 1);
            g->AddControlEdge(*primary_init_node, init_node);
            g->AddControlEdge(init_node, cur_init_op);
            // init_value
            const Edge* init_value_edge = nullptr;
            TF_RETURN_IF_ERROR(target_node->input_edge(2, &init_value_edge));
            auto* init_value_node =
                CopyNode(g, init_value_edge->src(), new_device_name, i);
            g->AddEdge(init_value_node, init_value_edge->src_output(),
                       init_node, 2);

            // empty_key
            const Edge* empty_key_edge = nullptr;
            TF_RETURN_IF_ERROR(target_node->input_edge(3, &empty_key_edge));
            Node* empty_key_node =
                CopyNode(g, empty_key_edge->src(), new_device_name, i);

            g->AddEdge(empty_key_node, empty_key_edge->src_output(), init_node,
                       3);
          }
          return Status::OK();
        }));
  }
  return Status::OK();
}

Status MakeEVFilterOp(Graph* g, PartIdToNodeMap& ev_node_vec,
                      std::vector<Node*>& filtered_node_vec,
                      std::vector<std::pair<Node*, Node*>>& primary_ev_filters,
                      std::unordered_set<Node*>* nodes_to_delete,
                      DataType& key_type, DataType& value_type,
                      int ev_partition_num, int i, int cur_partition_num) {
  auto* ev_node = ev_node_vec[i];
  auto* primary_ev_filter_node = primary_ev_filters[i].first;
  Status s;
  if (i < ev_partition_num) {
    for (auto* o_node : ev_node->out_nodes()) {
      if (o_node->type_string() == ::des::kEvExportOp) {
        TF_RETURN_IF_ERROR(GetNodeAttr(o_node->attrs(), "Tkeys", &key_type));
        TF_RETURN_IF_ERROR(GetNodeAttr(o_node->attrs(), "dtype", &value_type));
        nodes_to_delete->emplace(o_node);
      }
    }
  }

  NodeDef filter_storage_node_def;
  TF_RETURN_IF_ERROR(
      NodeDefBuilder("FilterStorage/" + ev_node->name(), ::des::kEvExportOp)
          .Input(ev_node->name(), 0, ev_node->output_type(0))
          .Input("partition_num", 0, DT_INT32)
          .Attr("partition_id", i)
          .Attr("partition_nums", cur_partition_num)
          .Attr("device_id", i)
          .Attr("Tkeys", key_type)
          .Attr("dtype", value_type)
          .Device(ev_node->assigned_device_name())
          .Finalize(&filter_storage_node_def));
  Node* filter_node = g->AddNode(filter_storage_node_def, &s);
  TF_RETURN_IF_ERROR(s);
  filter_node->set_assigned_device_name(ev_node->assigned_device_name());
  filtered_node_vec.push_back(filter_node);
  if (primary_ev_filter_node == nullptr) {
    primary_ev_filters[i].first = filter_node;
  } else {
    g->AddControlEdge(filter_node, primary_ev_filter_node);
  }
  return s;
}

Status MakeEVMulImportOp(
    Graph* g, Node* import_op_main, const Node* ev_node,
    const std::vector<Node*>& filtered_node_vec,
    std::vector<std::pair<Node*, Node*>>& primary_ev_filters,
    std::unordered_set<Node*>* nodes_to_delete, DataType key_type,
    DataType value_type, int ev_partition_num, int i, int cur_partition_num) {
  Status s;
  std::vector<Node*> sorted_filted_vec{filtered_node_vec[i]};
  for (int j = 0; j < filtered_node_vec.size(); ++j) {
    auto* filter_node = filtered_node_vec[j];
    if ((i != j) && (filter_node != nullptr)) {
      sorted_filted_vec.emplace_back(filter_node);
    }
  }
  auto* primary_ev_import_node = primary_ev_filters[i].second;
  std::vector<NodeDefBuilder::NodeOut> import_keys;
  std::vector<NodeDefBuilder::NodeOut> import_values;
  std::vector<NodeDefBuilder::NodeOut> import_versions;
  std::vector<NodeDefBuilder::NodeOut> import_freqs;
  for (int j = 0; j < sorted_filted_vec.size(); ++j) {
    import_keys.emplace_back(sorted_filted_vec[j]->name(), i,
                             sorted_filted_vec[j]->output_type(0));
    import_values.emplace_back(sorted_filted_vec[j]->name(),
                               cur_partition_num + i, DT_FLOAT);
    import_versions.emplace_back(sorted_filted_vec[j]->name(),
                                 2 * cur_partition_num + i, DT_INT64);
    import_freqs.emplace_back(
        sorted_filted_vec[j]->name(), 3 * cur_partition_num + i,
        sorted_filted_vec[j]->output_type(3 * cur_partition_num + i));
  }
  NodeDef import_storage_node_def;
  TF_RETURN_IF_ERROR(
      NodeDefBuilder(::des::kEvImportOp + ev_node->name(), ::des::kEvImportOp)
          .Input(ev_node->name(), 0, ev_node->output_type(0))
          .Input("partition_num", 0, DT_INT32)
          .Input(import_keys)
          .Input(import_values)
          .Input(import_versions)
          .Input(import_freqs)
          .Attr("partition_id", i)
          .Attr("partition_nums", cur_partition_num)
          .Attr("device_id", i)
          .Attr("Tkeys", key_type)
          .Attr("dtype", value_type)
          .Device(ev_node->assigned_device_name())
          .Finalize(&import_storage_node_def));
  Node* import_node = g->AddNode(import_storage_node_def, &s);
  TF_RETURN_IF_ERROR(s);
  import_node->set_assigned_device_name(ev_node->assigned_device_name());
  g->AddControlEdge(import_node, import_op_main);
  if (primary_ev_import_node == nullptr) {
    primary_ev_filters[i].second = import_node;
  } else {
    g->AddControlEdge(primary_ev_import_node, import_node);
  }
  if (i < ev_partition_num) {
    for (auto* o_node : ev_node->out_nodes()) {
      if (o_node->type_string() == ::des::kEvImportOp) {
        nodes_to_delete->insert(o_node);
      }
    }
  }
  return s;
}

Status InitDynamicPartitionGraphMetaEVUtil(
    const Node* ev_node,
    std::unordered_map<std::string, DynamicPartitionSubGraph>&
        prefix_name_map) {
  const string gather_name = "KvResourceGather";
  for (auto* o_node : ev_node->out_nodes()) {
    if (o_node->type_string() == gather_name) {
      std::string prefix_name = gather_name.substr(0, gather_name.rfind("/"));
      auto it = prefix_name_map.find(prefix_name);
      if (it == prefix_name_map.end()) {
        prefix_name_map.emplace(prefix_name, DynamicPartitionSubGraph());
      }
      auto& node_name_meta = prefix_name_map[prefix_name];
      node_name_meta.gather_node_vec.emplace_back(o_node);
      const Edge* input_edge = nullptr;
      TF_RETURN_IF_ERROR(o_node->input_edge(1, &input_edge));
      if (input_edge->src()->type_string() == kDynamicPartition) {
        node_name_meta.dynamic_partition_node = input_edge->src();
      }

      for (auto* oo_node : o_node->out_nodes()) {
        if (oo_node->type_string() == kIdentityOp) {
          node_name_meta.identity_node_vec.emplace_back(oo_node);
          for (auto* ooo_node : oo_node->out_nodes()) {
            if (ooo_node->type_string() == kParaDynamicStitch) {
              node_name_meta.dynamic_stitch_node = ooo_node;
            }
          }
        }
      }
    }
  }
  return Status::OK();
}

Status InitDynamicPartitionGraphMetaForEVForScalingDown(
    PartIdToNodeMap& ev_node_vec,
    std::unordered_map<std::string, DynamicPartitionSubGraph>& prefix_name_map,
    const std::vector<bool>& part_to_scale_down) {
  for (int i = 0; i < part_to_scale_down.size(); ++i) {
    auto* ev_node = ev_node_vec[i];
    if (!part_to_scale_down[i]) {
      TF_RETURN_IF_ERROR(
          InitDynamicPartitionGraphMetaEVUtil(ev_node, prefix_name_map));
    }
  }
  return Status::OK();
}

Status InitDynamicPartitionGraphMetaForEVForScalingUp(
    int part_num, PartIdToNodeMap& ev_node_vec,
    std::unordered_map<std::string, DynamicPartitionSubGraph>&
        prefix_name_map) {
  for (int i = 0; i < part_num; ++i) {
    auto* ev_node = ev_node_vec[i];
    TF_RETURN_IF_ERROR(
        InitDynamicPartitionGraphMetaEVUtil(ev_node, prefix_name_map));
  }
  return Status::OK();
}

Status InitDynamicPartitionGraphMetaVarUtil(
    const Node* ev_node, int i, 
    std::unordered_map<std::string, DynamicPartitionSubGraph>&
        prefix_name_map) {
  string gather_name = "GatherV2";
  for (auto* o_node : ev_node->out_nodes()) {
    if (o_node->type_string() == kIdentityOp) {
      for (auto* oo_node : o_node->out_nodes()) {
        if (oo_node->type_string() == gather_name) {
          std::string prefix_name =
              oo_node->name().substr(0, oo_node->name().rfind("/"));
          auto it = prefix_name_map.find(prefix_name);
          if (it == prefix_name_map.end()) {
            prefix_name_map.emplace(prefix_name, DynamicPartitionSubGraph());
          }
          auto& node_name_meta = prefix_name_map[prefix_name];
          node_name_meta.gather_node_vec.emplace_back(oo_node);
          node_name_meta.identity_node_vec.emplace_back(
              oo_node);  // no extra identity
          if (i == 0) {
            const Edge* input_edge = nullptr;
            TF_RETURN_IF_ERROR(oo_node->input_edge(1, &input_edge));
            if (input_edge->src()->type_string() == kDynamicPartition) {
              node_name_meta.dynamic_partition_node = input_edge->src();
            }

            for (auto* ooo_node : oo_node->out_nodes()) {
              if (ooo_node->type_string() == kParaDynamicStitch) {
                node_name_meta.dynamic_stitch_node = ooo_node;
              }
            }
          }
        }
      }
    }
  }
  return Status::OK();
}

Status InitDynamicPartitionGraphMetaForVarForScalingDown(
    PartIdToNodeMap& ev_node_vec,
    std::unordered_map<std::string, DynamicPartitionSubGraph>& prefix_name_map,
    const std::vector<bool>& part_to_scale_down) {
  for (int i = 0; i < part_to_scale_down.size(); ++i) {
    auto* ev_node = ev_node_vec[i];
    if (!part_to_scale_down[i]) {
      TF_RETURN_IF_ERROR(
          InitDynamicPartitionGraphMetaVarUtil(ev_node, i, prefix_name_map));
    }
  }
  return Status::OK();
}

Status InitDynamicPartitionGraphMetaForVarForScalingUp(
    int part_num, PartIdToNodeMap& ev_node_vec,
    std::unordered_map<std::string, DynamicPartitionSubGraph>&
        prefix_name_map) {
  for (int i = 0; i < part_num; ++i) {
    auto* ev_node = ev_node_vec[i];
    TF_RETURN_IF_ERROR(
        InitDynamicPartitionGraphMetaVarUtil(ev_node, i, prefix_name_map));
  }
  return Status::OK();
}

Status InitDynamicPartitionGraphMetaResVarUtil(
    const Node* ev_node,
    std::unordered_map<std::string, DynamicPartitionSubGraph>&
        prefix_name_map) {
  string gather_name = "ResourceGather";
  for (auto* o_node : ev_node->out_nodes()) {
    if (o_node->type_string() == gather_name) {
      std::string prefix_name = gather_name.substr(0, gather_name.rfind("/"));
      auto it = prefix_name_map.find(prefix_name);
      if (it == prefix_name_map.end()) {
        prefix_name_map.emplace(prefix_name, DynamicPartitionSubGraph());
      }
      auto& node_name_meta = prefix_name_map[prefix_name];
      node_name_meta.gather_node_vec.emplace_back(o_node);
      const Edge* input_edge = nullptr;
      TF_RETURN_IF_ERROR(o_node->input_edge(1, &input_edge));
      if (input_edge->src()->type_string() == kDynamicPartition) {
        node_name_meta.dynamic_partition_node = input_edge->src();
      }

      for (auto* oo_node : o_node->out_nodes()) {
        if (oo_node->type_string() == kIdentityOp) {
          node_name_meta.identity_node_vec.emplace_back(oo_node);
          for (auto* ooo_node : oo_node->out_nodes()) {
            if (ooo_node->type_string() == kParaDynamicStitch) {
              node_name_meta.dynamic_stitch_node = ooo_node;
            }
          }
        }
      }
    }
  }
  return Status::OK();
}

Status InitDynamicPartitionGraphMetaForResVarForScalingDown(
    PartIdToNodeMap& ev_node_vec,
    std::unordered_map<std::string, DynamicPartitionSubGraph>& prefix_name_map,
    const std::vector<bool>& part_to_scale_down) {
  for (int i = 0; i < part_to_scale_down.size(); ++i) {
    auto* ev_node = ev_node_vec[i];
    if (!part_to_scale_down[i]) {
      TF_RETURN_IF_ERROR(
          InitDynamicPartitionGraphMetaResVarUtil(ev_node, prefix_name_map));
    }
  }
  return Status::OK();
}

Status InitDynamicPartitionGraphMetaForResVarForScalingUp(
    int part_num, PartIdToNodeMap& ev_node_vec,
    std::unordered_map<std::string, DynamicPartitionSubGraph>&
        prefix_name_map) {
  for (int i = 0; i < part_num; ++i) {
    auto* ev_node = ev_node_vec[i];
    TF_RETURN_IF_ERROR(
        InitDynamicPartitionGraphMetaResVarUtil(ev_node, prefix_name_map));
  }
  return Status::OK();
}

Status InitDynamicPartitionGraphMeta(
    const VarType& var_type, int part_num, PartIdToNodeMap& ev_node_vec,
    std::unordered_map<std::string, DynamicPartitionSubGraph>& prefix_name_map,
    const std::vector<bool>* part_to_scale_down = nullptr) {
  switch (var_type) {
    case VarType::EMBEDDING_VAR: {
      if (part_to_scale_down == nullptr) {
        TF_RETURN_IF_ERROR(InitDynamicPartitionGraphMetaForEVForScalingUp(
            part_num, ev_node_vec, prefix_name_map));
      } else {
        TF_RETURN_IF_ERROR(InitDynamicPartitionGraphMetaForEVForScalingDown(
            ev_node_vec, prefix_name_map, *part_to_scale_down));
      }

      break;
    }
    case VarType::REF_VAR: {
      if (part_to_scale_down == nullptr) {
        TF_RETURN_IF_ERROR(InitDynamicPartitionGraphMetaForVarForScalingUp(
            part_num, ev_node_vec, prefix_name_map));
      } else {
        TF_RETURN_IF_ERROR(InitDynamicPartitionGraphMetaForVarForScalingDown(
            ev_node_vec, prefix_name_map, *part_to_scale_down));
      }
      break;
    }
    case VarType::RESOURCE_VAR: {
      if (part_to_scale_down == nullptr) {
        TF_RETURN_IF_ERROR(InitDynamicPartitionGraphMetaForResVarForScalingUp(
            part_num, ev_node_vec, prefix_name_map));
      } else {
        TF_RETURN_IF_ERROR(InitDynamicPartitionGraphMetaForResVarForScalingDown(
            ev_node_vec, prefix_name_map, *part_to_scale_down));
      }
      break;
    }
    default:  // DENSE_LAYER_VAR
      return Status::OK();
  }

  return Status::OK();
}

Status MakeDynamicPartitionOp(
    const VarType& var_type, int part_num,
    DynamicPartitionSubGraph& dynamic_partition_sub_graph, Graph* g,
    std::unordered_set<Node*>* nodes_to_delete,
    const std::vector<bool>* part_to_scale_down = nullptr) {
  Status s;
  const Node* dynamic_partition_node =
      dynamic_partition_sub_graph.dynamic_partition_node;
  const std::vector<Node*>& gather_node_vec =
      dynamic_partition_sub_graph.gather_node_vec;

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
      break;
    }
    case VarType::REF_VAR: {
      Node* maximum_node = nullptr;
      TF_RETURN_IF_ERROR(input_edge->src()->input_node(0, &maximum_node));
      if (input_edge->src()->type_string() != "Cast") {
        // Cast -> Mod
        Node* divisor_node = nullptr;
        TF_RETURN_IF_ERROR(maximum_node->input_node(1, &divisor_node));
        Tensor new_part_nums(DT_INT64, TensorShape({}));
        new_part_nums.flat<int64>()(0) = part_num;
        divisor_node->ClearAttr("value");
        divisor_node->AddAttr("value", new_part_nums);
      } else {
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

      break;
    }
    case VarType::RESOURCE_VAR:
    case VarType::DENSE_RESOUCE_VAR:
    case VarType::DENSE_REF_VAR: {
      return Status::OK();
    }
  }

  for (auto* o_node : input_edge->src()->out_nodes()) {
    if (o_node->type_string() == kDynamicPartition) {
      const Edge* data_input_edge = nullptr;
      TF_RETURN_IF_ERROR(o_node->input_edge(0, &data_input_edge));
      if (data_input_edge->src()->type_string() != "Range") {  // ID
        NodeDef dp_node_def;
        Status s;
        TF_RETURN_IF_ERROR(
            NodeDefBuilder("DES/" + dynamic_partition_node->name(),
                           kDynamicPartition)
                .Input(data_input_edge->src()->name(),
                       data_input_edge->src_output(), DT_INT64)
                .Input(input_edge->src()->name(), input_edge->src_output(),
                       DT_INT32)
                .Attr("num_partitions", part_num)
                .Attr("T", DT_INT64)
                .Device(dynamic_partition_node->assigned_device_name())
                .Finalize(&dp_node_def));
        dynamic_partition_sub_graph.data_dp_node = g->AddNode(dp_node_def, &s);
        TF_RETURN_IF_ERROR(s);
        dynamic_partition_sub_graph.data_dp_node->set_assigned_device_name(
            dynamic_partition_node->assigned_device_name());
        g->AddEdge(data_input_edge->src(), data_input_edge->src_output(),
                   dynamic_partition_sub_graph.data_dp_node, 0);
        g->AddEdge(input_edge->src(), input_edge->src_output(),
                   dynamic_partition_sub_graph.data_dp_node, 1);
        nodes_to_delete->insert(o_node);
      } else {  // Indices
        NodeDef dp_node_def;
        Status s;
        TF_RETURN_IF_ERROR(
            NodeDefBuilder("DES/" + o_node->name(), kDynamicPartition)
                .Input(data_input_edge->src()->name(),
                       data_input_edge->src_output(), DT_INT32)
                .Input(input_edge->src()->name(), input_edge->src_output(),
                       DT_INT32)
                .Attr("num_partitions", part_num)
                .Attr("T", DT_INT32)
                .Device(o_node->assigned_device_name())
                .Finalize(&dp_node_def));
        dynamic_partition_sub_graph.indices_dp_node =
            g->AddNode(dp_node_def, &s);
        TF_RETURN_IF_ERROR(s);
        dynamic_partition_sub_graph.indices_dp_node->set_assigned_device_name(
            o_node->assigned_device_name());
        g->AddEdge(data_input_edge->src(), data_input_edge->src_output(),
                   dynamic_partition_sub_graph.indices_dp_node, 0);
        g->AddEdge(input_edge->src(), input_edge->src_output(),
                   dynamic_partition_sub_graph.indices_dp_node, 1);
        nodes_to_delete->insert(o_node);
      }
    }
  }
  if (num_partitions > part_num) {
    int offset = 0;
    for (int i = 0; i < part_to_scale_down->size(); ++i) {
      if (!(*part_to_scale_down)[i]) {
        g->AddEdge(dynamic_partition_sub_graph.data_dp_node,
                    i-offset, gather_node_vec[i-offset], 1);
      } else {
        offset++;
      }
    }
  } else {
    for (int i = 0; i < gather_node_vec.size(); ++i) {
      if (i < num_partitions) {
        TF_RETURN_IF_ERROR(
            g->UpdateEdge(dynamic_partition_sub_graph.data_dp_node, i,
                          gather_node_vec[i], 1));
      } else {
        g->AddEdge(dynamic_partition_sub_graph.data_dp_node, i,
                   gather_node_vec[i], 1);
      }
    }
  }

  return s;
}

Status MakeResourceGatherOp(Graph* g, Node* ori_ev_node, Node* new_ev_node,
                            const string& new_device_name, int i) {
  for (auto* target_node : ori_ev_node->out_nodes()) {
    if (target_node->type_string() == "KvResourceGather") {
      Node* gather_op = CopyNode(g, target_node, new_device_name, i);
      g->AddEdge(new_ev_node, 0, gather_op, 0);
      const Edge* gather_id_edge = nullptr;
      TF_RETURN_IF_ERROR(target_node->input_edge(1, &gather_id_edge));
      g->AddEdge(gather_id_edge->src(), gather_id_edge->src_output(), gather_op,
                 1);
      Node* old_default_value;
      TF_RETURN_IF_ERROR(target_node->input_node(2, &old_default_value));
      if (old_default_value->type_string() == "Const") {
        Node* new_default_value =
            CopyNode(g, old_default_value, new_device_name, i);
        g->AddEdge(new_default_value, 0, gather_op, 2);
      } else {
        TF_RETURN_IF_ERROR(
            MakeTruncatedNormalInitializer(g, old_default_value, gather_op, i));
      }

      for (auto* o_edge : target_node->out_edges()) {
        if (o_edge->dst()->type_string() == kIdentityOp) {
          Node* identity_op = CopyNode(g, o_edge->dst(), new_device_name, i);
          g->AddEdge(gather_op, 0, identity_op, 0);
        }
      }
    }
  }
  return Status::OK();
}

Status MakeDynamicStitchOp(
    DynamicPartitionSubGraph& dynamic_partition_sub_graph, int part_num,
    Graph* g, std::unordered_set<Node*>* nodes_to_delete,
    const std::vector<bool>* part_to_scale_down = nullptr) {
  DataType key_type;
  std::vector<NodeDefBuilder::NodeOut> indices_inputs;
  std::vector<NodeDefBuilder::NodeOut> identity_inputs;
  const std::vector<Node*>& identity_node_vec =
      dynamic_partition_sub_graph.identity_node_vec;
  for (int i = 0; i < identity_node_vec.size(); ++i) {
    indices_inputs.emplace_back(
        dynamic_partition_sub_graph.indices_dp_node->name(), i, DT_INT32);
    identity_inputs.emplace_back(identity_node_vec[i]->name(), 0,
                                 identity_node_vec[i]->output_type(0));
  }
  Node* dynamic_stitch_node = dynamic_partition_sub_graph.dynamic_stitch_node;
  TF_RETURN_IF_ERROR(GetNodeAttr(dynamic_stitch_node->attrs(), "T", &key_type));

  NodeDef ds_node_def;
  Status s;
  TF_RETURN_IF_ERROR(
      NodeDefBuilder("DES/" + dynamic_stitch_node->name(), kParaDynamicStitch)
          .Input(indices_inputs)
          .Input(identity_inputs)
          .Attr("N", part_num)
          .Attr("T", key_type)
          .Device(dynamic_stitch_node->assigned_device_name())
          .Finalize(&ds_node_def));
  dynamic_partition_sub_graph.new_dynamic_stitch_node =
      g->AddNode(ds_node_def, &s);
  TF_RETURN_IF_ERROR(s);
  dynamic_partition_sub_graph.new_dynamic_stitch_node->set_assigned_device_name(
      dynamic_stitch_node->assigned_device_name());

  for (auto* o_edge : dynamic_stitch_node->out_edges()) {
    if (o_edge->IsControlEdge()) {
      g->AddControlEdge(dynamic_partition_sub_graph.new_dynamic_stitch_node,
                        o_edge->dst());
    } else {
      g->AddEdge(dynamic_partition_sub_graph.new_dynamic_stitch_node,
                 o_edge->src_output(), o_edge->dst(), o_edge->dst_input());
    }
  }
  for (int i = 0; i < identity_node_vec.size(); ++i) {
    g->AddEdge(dynamic_partition_sub_graph.indices_dp_node, i,
               dynamic_partition_sub_graph.new_dynamic_stitch_node, i);
    g->AddEdge(identity_node_vec[i], 0,
               dynamic_partition_sub_graph.new_dynamic_stitch_node,
               part_num + i);
  }
  nodes_to_delete->insert(dynamic_stitch_node);
  return s;
}

Status UpdateOldBackWardGraph(
    const VarType& var_type, Node* data_dp_node, Node* indices_dp_node,
    const std::vector<std::string>& opt_ev_names, Graph* g, int i, int part_num,
    Node** concat_node, Node** axis_node, DataType& concat_type, bool is_shared,
    const std::string& prefix_name, std::vector<Node*>& concat_inputs,
    const Node* variable_node) {
  for (auto* node : variable_node->out_nodes()) {
    if (IsApplyNode(var_type, node)) {
      for (int j = node->num_inputs() - 1; j > opt_ev_names.size(); --j) {
        Node* i_node;
        TF_RETURN_IF_ERROR(node->input_node(j, &i_node));
        if (IsUnique(i_node)) {
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
    int cur_partition_nums, bool is_shared, const std::string& prefix_name,
    std::vector<Node*>& concat_inputs) {
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
    if (IsUnique(i_node)) {
      new_unique = CopyNode(g, i_node, new_device_name, i);
      g->AddEdge(new_unique, 0, new_apply_node, j);
      // unique INPUT 0
      Node* reshape_id;
      TF_RETURN_IF_ERROR(i_node->input_node(0, &reshape_id));
      Node* new_reshape_id = nullptr;
      if (is_shared) {
        for (auto* reshape_node : reshape_id->in_nodes()) {
          if (reshape_node->name().find(prefix_name) != -1) {
            new_reshape_id = CopyNode(g, reshape_node,
                                      reshape_node->assigned_device_name(), i);
            concat_inputs.emplace_back(new_reshape_id);
            break;
          }
        }
      } else {
        new_reshape_id =
            CopyNode(g, reshape_id, reshape_id->assigned_device_name(), i);
        g->AddEdge(new_reshape_id, 0, new_unique, 0);
      }

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
        for (auto* i_edge : axis_1->in_edges()) {
          if (i_edge->IsControlEdge()) {
            g->AddControlEdge(i_edge->src(), new_axis_1);
          }
        }
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

Status ChangeVarShape(Node* shape_node, int part_var_full_shape,
                      int cur_partition_nums, int i, const string& attr_name) {
  if (attr_name == "init_value") {
    Tensor old_shape_tensor;
    TF_RETURN_IF_ERROR(
        GetNodeAttr(shape_node->attrs(), attr_name, &old_shape_tensor));
    int tensor_size = old_shape_tensor.NumElements();
    int partition_shape = part_var_full_shape / cur_partition_nums;
    int remainder = part_var_full_shape % cur_partition_nums;
    if (remainder > i) {
      partition_shape += 1;
    }
    int new_part_shape = partition_shape;
    Tensor shape_tensor;
    TensorProto tensor_shape_proto;
    tensor_shape_proto.set_dtype(DT_FLOAT);
    TensorShape({tensor_size})
        .AsProto(tensor_shape_proto.mutable_tensor_shape());
    tensor_shape_proto.add_float_val(new_part_shape);
    if (tensor_size > 1) {
      for (int j = 1; j < tensor_size; ++j) {
        tensor_shape_proto.add_float_val(old_shape_tensor.flat<float>()(j));
      }
    }
    bool ret = shape_tensor.FromProto(tensor_shape_proto);
    if (!ret) return errors::Internal("shape tensor init error");
    shape_node->ClearAttr(attr_name);
    shape_node->AddAttr(attr_name, shape_tensor);
  } else if (attr_name == "value") {
    Tensor old_shape_tensor;
    TF_RETURN_IF_ERROR(
        GetNodeAttr(shape_node->attrs(), attr_name, &old_shape_tensor));
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
    shape_node->ClearAttr(attr_name);
    shape_node->AddAttr(attr_name, shape_tensor);
  } else {
    TensorShape old_shape;
    TF_RETURN_IF_ERROR(GetNodeAttr(shape_node->attrs(), attr_name, &old_shape));
    int tensor_size = old_shape.dims();
    int partition_shape = part_var_full_shape / cur_partition_nums;
    int remainder = part_var_full_shape % cur_partition_nums;
    if (remainder > i) {
      partition_shape += 1;
    }
    int new_part_shape = partition_shape;

    TensorShape new_shape = TensorShape({new_part_shape});
    if (tensor_size > 1) {
      for (int j = 1; j < tensor_size; ++j) {
        new_shape.AddDim(old_shape.dim_size(j));
      }
    }
    shape_node->ClearAttr(attr_name);
    shape_node->AddAttr(attr_name, new_shape);
  }
  return Status::OK();
}

Status MakeConcatOffsetOp(Graph* g, Node* grad_concat_node,
                          Node* grad_axis_node,
                          std::vector<Node*>& concat_nodes,
                          std::vector<Node*> concat_outputs,
                          int cur_partition_num) {
  if ((grad_concat_node == nullptr) || (grad_axis_node == nullptr)) {
    LOG(ERROR) << "WARNING: "
               << "grad_concat_node or grad_axis_node is nullptr";
    return Status::OK();
  }
  NodeDef concat_node_def;
  Status s;
  std::vector<NodeDefBuilder::NodeOut> concat_inputs;
  for (auto* cinputs : concat_nodes) {
    concat_inputs.emplace_back(cinputs->name(), 0, cinputs->output_type(0));
  }
  TF_RETURN_IF_ERROR(
      NodeDefBuilder("DES/" + grad_concat_node->name(), "ConcatOffset")
          .Input(grad_axis_node->name(), 0, grad_axis_node->output_type(0))
          .Input(concat_inputs)
          .Attr("N", cur_partition_num)
          .Device(grad_concat_node->assigned_device_name())
          .Finalize(&concat_node_def));
  Node* new_concat_node = g->AddNode(concat_node_def, &s);
  TF_RETURN_IF_ERROR(s);
  new_concat_node->set_assigned_device_name(
      grad_concat_node->assigned_device_name());
  g->AddEdge(grad_axis_node, 0, new_concat_node, 0);
  for (int i = 0; i < concat_nodes.size(); ++i) {
    g->AddEdge(concat_nodes[i], 0, new_concat_node, 1 + i);
  }
  for (int i = 0; i < concat_outputs.size(); ++i) {
    TF_RETURN_IF_ERROR(g->UpdateEdge(new_concat_node, i, concat_outputs[i], 1));
  }
  g->RemoveNode(grad_concat_node);
  return s;
}

Status MakeConcatOp(Graph* g, Node* concat_node,
                    std::vector<Node*>& concat_inputs, Node* axis_node,
                    DataType concat_type, int cur_partition_num) {
  if (concat_node != nullptr) {
    NodeDef concat_node_def;
    std::vector<NodeDefBuilder::NodeOut> concat_node_outs;
    for (auto* cinput : concat_inputs) {
      concat_node_outs.emplace_back(cinput->name(), 0, cinput->output_type(0));
    }
    Status s;
    TF_RETURN_IF_ERROR(
        NodeDefBuilder("DES/" + concat_node->name(), "ConcatV2")
            .Input(concat_node_outs)
            .Input(axis_node->name(), 0, axis_node->output_type(0))
            .Attr("N", cur_partition_num)
            .Attr("T", concat_type)
            .Device(concat_node->assigned_device_name())
            .Finalize(&concat_node_def));
    Node* new_concat_node = g->AddNode(concat_node_def, &s);
    TF_RETURN_IF_ERROR(s);
    new_concat_node->set_assigned_device_name(
        concat_node->assigned_device_name());
    for (int i = 0; i < concat_inputs.size(); ++i) {
      g->AddEdge(concat_inputs[i], 0, new_concat_node, i);
    }
    g->AddEdge(axis_node, 0, new_concat_node, concat_inputs.size());

    for (auto* o_edge : concat_node->out_edges()) {
      // TF_RETURN_IF_ERROR(g->UpdateEdge(new_concat_node, o_edge->src_output(),
      //                                  o_edge->dst(), o_edge->dst_input()));
      g->AddEdge(new_concat_node, o_edge->src_output(), o_edge->dst(),
                 o_edge->dst_input());
    }
    g->RemoveNode(concat_node);
  }
  return Status::OK();
}

Status MakePackOp(Graph* g, std::vector<Node*>& pack_inputs, Node* pack_node,
                  DataType concat_type, int axis, int cur_partition_num) {
  NodeDef concat_node_def;
  Status s;
  if (pack_node == nullptr) {
    return errors::Internal("pack_node is nullptr");
  }
  std::vector<NodeDefBuilder::NodeOut> pack_inputs_out;
  for (auto* pinputs : pack_inputs) {
    pack_inputs_out.emplace_back(pinputs->name(), 0, pinputs->output_type(0));
  }
  TF_RETURN_IF_ERROR(NodeDefBuilder("DES/" + pack_node->name(), "Pack")
                         .Input(pack_inputs_out)
                         .Attr("N", cur_partition_num)
                         .Attr("T", concat_type)
                         .Attr("axis", axis)
                         .Device(pack_node->assigned_device_name())
                         .Finalize(&concat_node_def));
  Node* new_concat_node = g->AddNode(concat_node_def, &s);
  TF_RETURN_IF_ERROR(s);
  new_concat_node->set_assigned_device_name(pack_node->assigned_device_name());
  for (int i = 0; i < pack_inputs.size(); ++i) {
    g->AddEdge(pack_inputs[i], 0, new_concat_node, i);
  }

  for (auto* i_edge : pack_node->in_edges()) {
    if (i_edge->IsControlEdge()) {
      g->AddControlEdge(i_edge->src(), new_concat_node);
    }
  }
  for (auto* o_edge : pack_node->out_edges()) {
    if (o_edge->IsControlEdge()) {
      g->AddControlEdge(new_concat_node, o_edge->dst());
    } else {
      TF_RETURN_IF_ERROR(g->UpdateEdge(new_concat_node, o_edge->src_output(),
                                       o_edge->dst(), o_edge->dst_input()));
    }
  }

  g->RemoveNode(pack_node);
  return s;
}

Status UpdateOldDenseBackWardGraph(
    VarType var_type, Graph* g, Node* dense_variable_node,
    Node** grad_concat_node, Node** grad_axis_node,
    std::vector<Node*>* concat_inputs, std::vector<Node*>* concat_outputs,
    int part_var_full_shape, int i, int cur_partition_nums,
    int ev_partition_num, const std::vector<std::string>& opt_ev_names,
    Node* cur_noop_node = nullptr) {
  for (auto* node : dense_variable_node->out_nodes()) {
    if (IsApplyNode(var_type, node)) {
      if (cur_noop_node != nullptr) {
        for (auto* edge : node->out_edges()) {
          for (auto* o_edge : edge->dst()->out_edges()) {
            if (o_edge->IsControlEdge()) {
              g->AddControlEdge(cur_noop_node, o_edge->dst());
            }
          }
        }
      }
      for (int j = node->num_inputs() - 1; j > opt_ev_names.size(); --j) {
        Node* i_node;
        TF_RETURN_IF_ERROR(node->input_node(j, &i_node));
        if (i_node->type_string() == "Identity") {
          Node* concat_grad_node;
          TF_RETURN_IF_ERROR(i_node->input_node(0, &concat_grad_node));
          if (concat_grad_node->type_string() == "Slice") {
            Node* shape_node;
            TF_RETURN_IF_ERROR(concat_grad_node->input_node(2, &shape_node));
            if (shape_node->type_string() == "ShapeN") {
              int shape_n;
              TF_RETURN_IF_ERROR(
                  GetNodeAttr(shape_node->attrs(), "N", &shape_n));
              int new_part_shape = part_var_full_shape / cur_partition_nums;
              if (part_var_full_shape % cur_partition_nums > i) {
                new_part_shape += 1;
              }
              shape_node->ClearAttr("N");
              shape_node->AddAttr("N", new_part_shape);
            } else {
              TF_RETURN_IF_ERROR(ChangeVarShape(shape_node, part_var_full_shape,
                                                cur_partition_nums, i,
                                                "value"));
            }

            const Edge* concat_offset_edge;
            TF_RETURN_IF_ERROR(
                concat_grad_node->input_edge(1, &concat_offset_edge));
            *grad_concat_node = concat_offset_edge->src();
            int part_num;
            TF_RETURN_IF_ERROR(GetNodeAttr(concat_offset_edge->src()->attrs(),
                                           "N", &part_num));
            TF_RETURN_IF_ERROR(
                concat_offset_edge->src()->input_node(0, grad_axis_node));

            concat_inputs->emplace_back(shape_node);
            concat_outputs->emplace_back(concat_grad_node);
          }
        }
      }
    }
  }
  return Status::OK();
}

Status MakeApplyOp(
    Graph* g, const Node* old_apply_node, Node* cur_noop_node,
    std::vector<Node*>* concat_inputs, std::vector<Node*>* concat_outputs,
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

  for (int j = old_apply_node->num_inputs() - 1; j > opt_ev_names.size(); --j) {
    Node* i_node;
    TF_RETURN_IF_ERROR(old_apply_node->input_node(j, &i_node));
    if (i_node->type_string() == "Identity") {
      Node* concat_grad_node;
      TF_RETURN_IF_ERROR(i_node->input_node(0, &concat_grad_node));
      if (concat_grad_node->type_string() == "Slice") {
        Node* new_grad_node = CopyNode(g, i_node, new_device_name, i);
        g->AddEdge(new_grad_node, 0, new_apply_node, j);

        Node* new_concat_grad_node = CopyNode(
            g, concat_grad_node, concat_grad_node->assigned_device_name(), i);
        g->AddEdge(new_concat_grad_node, 0, new_grad_node, 0);

        Node* prev_grad_node;
        TF_RETURN_IF_ERROR(concat_grad_node->input_node(0, &prev_grad_node));
        g->AddEdge(prev_grad_node, 0, new_concat_grad_node, 0);

        Node* concat_offset_node;
        TF_RETURN_IF_ERROR(
            concat_grad_node->input_node(1, &concat_offset_node));

        g->AddEdge(concat_offset_node, i, new_concat_grad_node, 1);
        Node* shape_node;
        TF_RETURN_IF_ERROR(concat_offset_node->input_node(1, &shape_node));
        Node* new_shape_node =
            CopyNode(g, shape_node, shape_node->assigned_device_name(), i);
        if (new_shape_node->type_string() == "ShapeN") {
          int shape_n;
          TF_RETURN_IF_ERROR(GetNodeAttr(shape_node->attrs(), "N", &shape_n));
          int new_part_shape = part_var_full_shape / cur_partition_nums;
          if (part_var_full_shape % cur_partition_nums > i) {
            new_part_shape += 1;
          }
          new_shape_node->ClearAttr("N");
          new_shape_node->AddAttr("N", new_part_shape);
        } else {
          Tensor old_shape_tensor;
          TF_RETURN_IF_ERROR(
              GetNodeAttr(shape_node->attrs(), "value", &old_shape_tensor));
          int tensor_size = old_shape_tensor.NumElements();
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
        }

        concat_inputs->emplace_back(new_shape_node);
        concat_outputs->emplace_back(new_concat_grad_node);
        g->AddEdge(new_shape_node, 0, new_concat_grad_node, 2);
        // TODO grad value size
        for (auto* i_edge : i_node->in_edges()) {
          if (i_edge->IsControlEdge()) {
            Node* control_node = i_edge->src();
            g->AddControlEdge(new_concat_grad_node, control_node);
            g->AddControlEdge(control_node, new_grad_node);
          }
        }
      } else {
        g->AddEdge(i_node, 0, new_apply_node, j);
      }
    } else {
      g->AddEdge(i_node, 0, new_apply_node, j);
    }
  }
  return Status::OK();
}

Status DeleteSparseBackWardGraph(VarType var_type, Node* cur_ev_node,
                                 const std::vector<string>& opt_ev_names,
                                 std::unordered_set<Node*>* nodes_to_delete) {
  for (auto* node : cur_ev_node->out_nodes()) {
    if (IsApplyNode(var_type, node)) {
      nodes_to_delete->insert(node);
      for (int j = node->num_inputs() - 1; j > opt_ev_names.size(); --j) {
        Node* i_node;
        TF_RETURN_IF_ERROR(node->input_node(j, &i_node));
        if (IsUnique(i_node)) {
          nodes_to_delete->insert(i_node);
          // unique INPUT 0
          Node* reshape_id;
          TF_RETURN_IF_ERROR(i_node->input_node(0, &reshape_id));
          nodes_to_delete->insert(reshape_id);

          for (auto* o_node : reshape_id->out_nodes()) {
            if (o_node->type_string() == "RecordSparseIndices") {
              nodes_to_delete->insert(o_node);
            }
          }

          Node* expand_dims;
          TF_RETURN_IF_ERROR(reshape_id->input_node(1, &expand_dims));
          nodes_to_delete->insert(expand_dims);

          // expand dims INPUT
          Node* expand_dims_size;
          TF_RETURN_IF_ERROR(expand_dims->input_node(0, &expand_dims_size));
          nodes_to_delete->insert(expand_dims_size);

          Node* expand_dims_dim;
          TF_RETURN_IF_ERROR(expand_dims->input_node(1, &expand_dims_dim));
          nodes_to_delete->insert(expand_dims_dim);

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
          nodes_to_delete->insert(i_node);
          // Input 0
          {
            Node* reshape;
            TF_RETURN_IF_ERROR(i_node->input_node(0, &reshape));
            nodes_to_delete->insert(reshape);
            // Reshape INPUT 0
            Node* control_denpency;
            TF_RETURN_IF_ERROR(reshape->input_node(0, &control_denpency));
            nodes_to_delete->insert(control_denpency);

            // control_dependency INPUT 0
            Node* gather_1;
            TF_RETURN_IF_ERROR(control_denpency->input_node(0, &gather_1));
            nodes_to_delete->insert(gather_1);

            // gather_1 INPUT2
            Node* axis_1;
            TF_RETURN_IF_ERROR(gather_1->input_node(2, &axis_1));
            nodes_to_delete->insert(axis_1);

            // Reshape INPUT 1
            Node* concat;
            TF_RETURN_IF_ERROR(reshape->input_node(1, &concat));
            nodes_to_delete->insert(concat);

            // concat INPUT 1
            Node* strided_slice;
            TF_RETURN_IF_ERROR(concat->input_node(1, &strided_slice));
            nodes_to_delete->insert(strided_slice);

            for (int k = 0; k < strided_slice->num_inputs(); ++k) {
              Node* partial_strided_slice;
              TF_RETURN_IF_ERROR(
                  strided_slice->input_node(k, &partial_strided_slice));
              nodes_to_delete->insert(partial_strided_slice);
            }

            // concat INPUT 2
            Node* axis;
            TF_RETURN_IF_ERROR(concat->input_node(2, &axis));
            nodes_to_delete->insert(axis);
          }

          // Input 2
          {
            Node* strided_slice;
            TF_RETURN_IF_ERROR(i_node->input_node(2, &strided_slice));
            nodes_to_delete->insert(strided_slice);

            Node* shape;
            TF_RETURN_IF_ERROR(strided_slice->input_node(0, &shape));
            nodes_to_delete->insert(shape);

            for (int k = 1; k < strided_slice->num_inputs(); ++k) {
              Node* partial_strided_slice;
              TF_RETURN_IF_ERROR(
                  strided_slice->input_node(k, &partial_strided_slice));
              nodes_to_delete->insert(partial_strided_slice);
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
                                std::unordered_set<Node*>* nodes_to_delete) {
  for (auto* node : cur_var_node->out_nodes()) {
    if (IsApplyNode(var_type, node)) {
      nodes_to_delete->insert(node);

      Node* i_node;
      TF_RETURN_IF_ERROR(node->input_node(node->num_inputs() - 1, &i_node));
      if (i_node->type_string() == "Identity") {
        nodes_to_delete->insert(i_node);

        Node* concat_grad_node;
        TF_RETURN_IF_ERROR(i_node->input_node(0, &concat_grad_node));
        nodes_to_delete->insert(concat_grad_node);

        Node* prev_grad_node;
        TF_RETURN_IF_ERROR(concat_grad_node->input_node(0, &prev_grad_node));

        Node* shape_node;
        TF_RETURN_IF_ERROR(concat_grad_node->input_node(2, &shape_node));
        nodes_to_delete->insert(shape_node);
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
      if (i >= partition_var.cur_partition_num) {
        continue;
      }
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

Status MakeTensorNameNode(Graph* g,
                          const std::vector<std::string>& tensor_names_vec,
                          const std::string& node_name,
                          const std::string& assigned_device_name, bool has_ev,
                          Node** tensor_name_node) {
  Status s;
  int tensor_size =
      has_ev ? tensor_names_vec.size() + 1 : tensor_names_vec.size();
  // tensor_names
  TensorProto tensor_name_proto;
  tensor_name_proto.set_dtype(DT_STRING);
  TensorShape({tensor_size}).AsProto(tensor_name_proto.mutable_tensor_shape());
  if (has_ev) {
    tensor_name_proto.add_string_val("global_step");
  }

  Tensor new_tensor_names;
  for (int j = 0; j < tensor_names_vec.size(); ++j) {
    tensor_name_proto.add_string_val(tensor_names_vec[j]);
  }
  bool ret = new_tensor_names.FromProto(tensor_name_proto);
  if (!ret) return errors::Internal("tensor_name tensor init error");
  NodeDef tensor_name_node_def;
  TF_RETURN_IF_ERROR(NodeDefBuilder(node_name, "Const")
                         .Attr("value", new_tensor_names)
                         .Attr("dtype", DT_STRING)
                         .Device(assigned_device_name)
                         .Finalize(&tensor_name_node_def));
  *tensor_name_node = g->AddNode(tensor_name_node_def, &s);
  (*tensor_name_node)->set_assigned_device_name(assigned_device_name);
  return s;
}

Status MakeShapeSlicsNode(
    Graph* g, int i, int part_num,
    const std::vector<std::string>& tensor_names_vec,
    const std::unordered_map<string, std::vector<int64>>& variable_shape,
    const std::string& node_name, const std::string& assigned_device_name,
    bool has_ev, Node** shape_and_slice_node) {
  int tensor_size =
      has_ev ? tensor_names_vec.size() + 1 : tensor_names_vec.size();
  Tensor new_tensor_shape;
  // tensor_names
  TensorProto tensor_shape_proto;
  tensor_shape_proto.set_dtype(DT_STRING);
  TensorShape({tensor_size}).AsProto(tensor_shape_proto.mutable_tensor_shape());
  if (has_ev) {
    tensor_shape_proto.add_string_val("");
  }
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
          int remainder = shape_and_slice[j] % part_num;
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

  bool ret = new_tensor_shape.FromProto(tensor_shape_proto);
  if (!ret) {
    return errors::Internal("tensor_name tensor init error");
  }
  Status s;
  NodeDef shape_slice_node_def;
  TF_RETURN_IF_ERROR(NodeDefBuilder(node_name, "Const")
                         .Attr("value", new_tensor_shape)
                         .Attr("dtype", DT_STRING)
                         .Device(assigned_device_name)
                         .Finalize(&shape_slice_node_def));
  *shape_and_slice_node = g->AddNode(shape_slice_node_def, &s);
  (*shape_and_slice_node)->set_assigned_device_name(assigned_device_name);
  return s;
}

Status MakeEvNamesNode(Graph* g, const std::vector<std::string>& ev_names_vec,
                       const std::string& node_name,
                       const std::string& assigned_device_name,
                       Node** ev_name_node) {
  // ev_names
  Status s;
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
  TF_RETURN_IF_ERROR(NodeDefBuilder(node_name, "Const")
                         .Attr("value", ev_names_tensor)
                         .Attr("dtype", DT_STRING)
                         .Device(assigned_device_name)
                         .Finalize(&ev_name_node_def));
  *ev_name_node = g->AddNode(ev_name_node_def, &s);
  (*ev_name_node)->set_assigned_device_name(assigned_device_name);
  return s;
}

Status MakeKvLookupResNode(
    Graph* g, const std::vector<Node*>& kv_lookup_resource_node_vec,
    std::vector<DataType>* ev_dtypes,
    const std::vector<DataType>& key_data_types, const std::string& node_name,
    const std::string& assigned_device_name, Node** kv_lookup_resource_node) {
  Status s;
  std::vector<NodeDefBuilder::NodeOut> kv_lookup_resource_input;
  for (auto* n : kv_lookup_resource_node_vec) {
    kv_lookup_resource_input.emplace_back(n->name(), 0, n->output_type(0));
    ev_dtypes->emplace_back(DT_INT64);
  }
  DataType key_type;
  if (key_data_types.size() == 0) {
    key_type = DT_INT64;
    Tensor const_tensor(DT_INT64, TensorShape({}));
    // ev_resources
    NodeDef const_node_def;
    TF_RETURN_IF_ERROR(NodeDefBuilder(node_name, "Const")
                           .Attr("dtype", key_type)
                           .Attr("value", const_tensor)
                           .Device(assigned_device_name)
                           .Finalize(&const_node_def));
    *kv_lookup_resource_node = g->AddNode(const_node_def, &s);
    (*kv_lookup_resource_node)->set_assigned_device_name(assigned_device_name);
  } else {
    key_type = key_data_types[0];
    // ev_resources
    NodeDef kv_lookup_resource_node_def;
    int n = kv_lookup_resource_node_vec.size();
    TF_RETURN_IF_ERROR(NodeDefBuilder(node_name, "Pack")
                           .Input(kv_lookup_resource_input)
                           .Attr("N", n)
                           .Attr("T", key_type)
                           .Attr("axis", 0)
                           .Device(assigned_device_name)
                           .Finalize(&kv_lookup_resource_node_def));
    *kv_lookup_resource_node = g->AddNode(kv_lookup_resource_node_def, &s);
    (*kv_lookup_resource_node)->set_assigned_device_name(assigned_device_name);
  }
  return s;
}

Status AddNewSaverGraph(
    Graph* g, bool& has_ev, Node** new_sharded_filename,
    std::vector<string>& tensor_names_vec, std::vector<DataType>& n_dtypes,
    std::vector<Node*>& pack_inputs, int& axis, Node** pack_node,
    std::unordered_map<std::string, PartitionedVariable>&
        primary_node_metas_map,
    std::unordered_map<string, std::vector<int64>>& variable_shape,
    std::vector<Node*>& restore_tensor_vec, std::string& assigned_device_name,
    std::vector<Node*>& save_node_vec, int i, int cur_partition_nums) {
  Node* ori_save_node = save_node_vec[0];
  std::vector<Node*> tensor_vec;
  const std::string& node_name_prefix =
      ori_save_node->name() + "_" + std::to_string(i);
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
  Node* sharded_filename;
  Node* tensor_name_node;
  Node* shape_slice_node;

  {
    TF_RETURN_IF_ERROR(ori_save_node->input_node(0, &sharded_filename));
    *new_sharded_filename =
        CopyNode(g, sharded_filename, assigned_device_name, i);
    pack_inputs.emplace_back(*new_sharded_filename);

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

  TF_RETURN_IF_ERROR(MakeTensorNameNode(
      g, tensor_names_vec, node_name_prefix + "/tensor_names",
      assigned_device_name, has_ev, &tensor_name_node));
  TF_RETURN_IF_ERROR(
      MakeShapeSlicsNode(g, i, cur_partition_nums, tensor_names_vec,
                         variable_shape, node_name_prefix + "/shape_and_slices",
                         assigned_device_name, has_ev, &shape_slice_node));

  Node* ev_name_node;
  Node* kv_lookup_resource_node;

  TF_RETURN_IF_ERROR(MakeEvNamesNode(g, ev_names_vec,
                                     node_name_prefix + "/ev_names",
                                     assigned_device_name, &ev_name_node));

  TF_RETURN_IF_ERROR(
      MakeKvLookupResNode(g, kv_lookup_resource_node_vec, &ev_dtypes,
                          key_data_types, node_name_prefix + "/ev_resources",
                          assigned_device_name, &kv_lookup_resource_node));

  if (n_dtypes.size() > 0) {
    // tensor_names
    NodeDef save_node_def;
    TF_RETURN_IF_ERROR(
        NodeDefBuilder(node_name_prefix, kSaveOp)
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
          }
        }
      }
    }
  }
  for (auto* o_node : sharded_filename->out_nodes()) {
    if (o_node->type_string() == "Pack") {
      TF_RETURN_IF_ERROR(GetNodeAttr(o_node->attrs(), "axis", &axis));
      *pack_node = o_node;
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
  std::string node_prefix = ori_restore_node->name() + "_" + std::to_string(i);
  Node* tensor_name_node;
  Node* shape_slice_node;
  TF_RETURN_IF_ERROR(
      MakeTensorNameNode(g, tensor_names_vec, node_prefix + "/tensor_names",
                         assigned_device_name, has_ev, &tensor_name_node));
  TF_RETURN_IF_ERROR(
      MakeShapeSlicsNode(g, i, cur_partition_nums, tensor_names_vec,
                         variable_shape, node_prefix + "/shape_and_slices",
                         assigned_device_name, has_ev, &shape_slice_node));

  NodeDef restore_node_def;
  if (has_ev) {
    n_dtypes.erase(n_dtypes.begin());
  }
  if (n_dtypes.size() > 0) {
    TF_RETURN_IF_ERROR(NodeDefBuilder(node_prefix, kRestoreOp)
                           .Input(new_sharded_filename->name(), 0,
                                  new_sharded_filename->output_type(0))
                           .Input(tensor_name_node->name(), 0,
                                  tensor_name_node->output_type(0))
                           .Input(shape_slice_node->name(), 0,
                                  shape_slice_node->output_type(0))
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

Status SavedModelInputHelper(
    const Node* input_node, const std::string& tensor_n,
    const std::unordered_map<std::string, Node*>& unpartitioned_node_map,
    std::unordered_set<Node*>& eval_nodes_to_add,
    std::unordered_map<std::string, PartitionedVariable>&
        primary_node_metas_map,
    std::unordered_map<string, string>& opt_to_primary_map, Node** actual_node,
    int& actual_part_id, int cur_partition_nums, bool scaling_up) {
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
        *actual_node = variable_node;
      }
    } else {  // RefVariable
      if (identity_node->name() != "global_step") {
        auto part_it = unpartitioned_node_map.find(identity_node->name());
        if (part_it != unpartitioned_node_map.end()) {
          eval_nodes_to_add.emplace(identity_node);
        } else {
          *actual_node = identity_node;
          int part_id = GetNodePartIndex(*actual_node);
          if ((part_id >= cur_partition_nums) && !scaling_up) {
            auto tmp_it = primary_node_metas_map.find(tensor_n);
            if (tmp_it != primary_node_metas_map.end()) {
              actual_part_id = tmp_it->second.node_device_map[*actual_node];
            } else if (opt_to_primary_map.find(tensor_n) !=
                       opt_to_primary_map.end()) {
              auto t_it =
                  primary_node_metas_map.find(opt_to_primary_map[tensor_n]);
              actual_part_id = t_it->second.node_device_map[*actual_node];
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
      *actual_node = read_variable_node;
    }
  } else if (input_node->type_string() == "KvVarHandleOp") {
    *actual_node = const_cast<Node*>(input_node);
  } else {  // moving_average
    auto part_it = unpartitioned_node_map.find(input_node->name());
    if (part_it != unpartitioned_node_map.end()) {
      eval_nodes_to_add.emplace(const_cast<Node*>(input_node));
    } else {
      *actual_node = const_cast<Node*>(input_node);
    }
  }
  return Status::OK();
}

Status FillTensorShape(
    const Node* tensor_names, const Node* shape_and_slices,
    const Node* ori_save_node,
    std::unordered_map<std::string, PartitionedVariable>&
        primary_node_metas_map,
    const std::unordered_map<std::string, Node*>& unpartitioned_node_map,
    std::unordered_map<string, std::vector<int64>>& variable_shape,
    std::unordered_map<string, string>& opt_to_primary_map,
    int cur_partition_nums, bool scaling_up,
    std::unordered_set<Node*>& eval_nodes_to_add,
    std::vector<string>* new_tensor_shape,
    std::vector<string>* new_tensor_name) {
  Tensor tensor_name_t;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(tensor_names->attrs(), "value", &tensor_name_t));
  Tensor shape_and_slice_t;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(shape_and_slices->attrs(), "value", &shape_and_slice_t));

  for (int k = 0; k < tensor_name_t.dim_size(0); ++k) {
    string tensor_name = tensor_name_t.flat<string>()(k);
    auto s_and_s_s = shape_and_slice_t.flat<string>()(k);
    new_tensor_name->emplace_back(tensor_name);
    Node* input_node = nullptr;
    TF_RETURN_IF_ERROR(
        ori_save_node->input_node(kSaverTensorInputOffset + k, &input_node));
    Node* actual_node = nullptr;
    int actual_part_id = 0;
    TF_RETURN_IF_ERROR(SavedModelInputHelper(
        input_node, tensor_name, unpartitioned_node_map, eval_nodes_to_add,
        primary_node_metas_map, opt_to_primary_map, &actual_node,
        actual_part_id, cur_partition_nums, scaling_up));
    auto it = variable_shape.find(tensor_name);
    if (it == variable_shape.end()) {
      // TODO (JUNQI): figure out which variable is this case
      new_tensor_shape->emplace_back(s_and_s_s);
    } else {
      int part_id = GetNodePartIndex(actual_node);
      if (part_id >= cur_partition_nums) {
        part_id = actual_part_id;
      }
      int var_partition_nums = 0;
      auto tmp_it = primary_node_metas_map.find(tensor_name);
      if (tmp_it != primary_node_metas_map.end()) {
        var_partition_nums = tmp_it->second.cur_partition_num;
      } else if (opt_to_primary_map.find(tensor_name) !=
                 opt_to_primary_map.end()) {
        auto t_it =
            primary_node_metas_map.find(opt_to_primary_map[tensor_name]);
        var_partition_nums = t_it->second.cur_partition_num;
      }
      if (var_partition_nums != cur_partition_nums) {
        new_tensor_shape->emplace_back(s_and_s_s);
        continue;
      }
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
          shape_and_slice[j + items.size()] = dim / var_partition_nums;
          remainder = dim % var_partition_nums;
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
      new_tensor_shape->emplace_back(tmp_shape_and_slice);
      variable_shape[tensor_name] = shape_and_slice;
    }
  }
  return Status::OK();
}

Status AddNodesToMove(
    Graph* g, int i, Node* ori_save_node,
    const std::unordered_map<Node*, std::pair<string, string>>& nodes_to_add,
    std::vector<string>* new_tensor_shape,
    std::vector<string>* new_tensor_name) {
  // place nodes on ps which is scaled out on ps:0
  if ((i == 0) && (nodes_to_add.size() > 0)) {
    int old_tensor_size = new_tensor_shape->size();
    std::vector<DataType> n_dtypes;
    TF_RETURN_IF_ERROR(
        GetNodeAttr(ori_save_node->attrs(), "dtypes", &n_dtypes));
    int k = 0;
    for (auto& it : nodes_to_add) {
      new_tensor_name->emplace_back(it.second.first);
      new_tensor_shape->emplace_back(it.second.second);
      if (it.second.first == "global_step") {
        n_dtypes.emplace_back(DT_INT64);
      } else {
        n_dtypes.emplace_back(DT_FLOAT);
      }
      g->AddEdge(it.first, 0, ori_save_node,
                 kSaverTensorInputOffset + old_tensor_size + k);
      ++k;
    }
    ori_save_node->ClearAttr("dtypes");
    ori_save_node->AddAttr("dtypes", n_dtypes);
  }
  return Status::OK();
}

Status MakeNewSaverInputNode(const std::string& node_name, int dst_input,
                             Graph* g, Node* cur_saver_node,
                             const std::vector<string>& new_tensor_name) {
  string assigned_device_name = cur_saver_node->assigned_device_name();
  int tensor_size = new_tensor_name.size();
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

  Status s;
  NodeDef name_node_def;
  TF_RETURN_IF_ERROR(NodeDefBuilder(node_name + "/Copy", "Const")
                         .Attr("value", new_tensor_name_t)
                         .Attr("dtype", DT_STRING)
                         .Device(assigned_device_name)
                         .Finalize(&name_node_def));
  Node* name_node = g->AddNode(name_node_def, &s);
  TF_RETURN_IF_ERROR(s);
  name_node->set_assigned_device_name(assigned_device_name);
  TF_RETURN_IF_ERROR(g->UpdateEdge(name_node, 0, cur_saver_node, dst_input));
  return s;
}

Status RewritePrevSaveGraph(
    Graph* g, int i, int cur_partition_nums, bool scaling_up,
    const std::vector<Node*>& save_node_vec, std::vector<Node*>& pack_inputs,
    DataType& concat_type,
    std::unordered_map<std::string, PartitionedVariable>&
        primary_node_metas_map,
    std::unordered_map<std::string, Node*>& unpartitioned_node_map,
    std::unordered_map<string, std::vector<int64>>& variable_shape,
    const std::unordered_map<Node*, std::pair<string, string>>& nodes_to_add,
    std::unordered_set<Node*>* nodes_to_delete,
    std::unordered_set<Node*>& eval_nodes_to_add,
    std::unordered_map<string, string>& opt_to_primary_map) {
  Node* cur_saver_node = save_node_vec[i];

  Node* sharded_filename;
  TF_RETURN_IF_ERROR(cur_saver_node->input_node(0, &sharded_filename));
  pack_inputs.emplace_back(sharded_filename);
  concat_type = sharded_filename->output_type(0);

  Node* tensor_names;
  TF_RETURN_IF_ERROR(cur_saver_node->input_node(1, &tensor_names));
  nodes_to_delete->insert(tensor_names);

  Node* shape_and_slices;
  TF_RETURN_IF_ERROR(cur_saver_node->input_node(2, &shape_and_slices));
  nodes_to_delete->insert(shape_and_slices);

  std::vector<string> new_tensor_shape;
  std::vector<string> new_tensor_name;
  TF_RETURN_IF_ERROR(FillTensorShape(
      tensor_names, shape_and_slices, cur_saver_node, primary_node_metas_map,
      unpartitioned_node_map, variable_shape, opt_to_primary_map,
      cur_partition_nums, scaling_up, eval_nodes_to_add, &new_tensor_shape,
      &new_tensor_name));

  TF_RETURN_IF_ERROR(AddNodesToMove(g, i, cur_saver_node, nodes_to_add,
                                    &new_tensor_shape, &new_tensor_name));
  TF_RETURN_IF_ERROR(MakeNewSaverInputNode(tensor_names->name(), 1, g,
                                           cur_saver_node, new_tensor_name));
  TF_RETURN_IF_ERROR(MakeNewSaverInputNode(shape_and_slices->name(), 2, g,
                                           cur_saver_node, new_tensor_shape));

  return Status::OK();
}

Status FindReserverdNode(
    const Tensor& tensor_name_t, const Tensor& shape_and_slice_t,
    const Node* cur_save_node,
    const std::unordered_map<string, std::vector<int64>>
        partitioned_variable_shape,
    const std::unordered_map<std::string, Node*>& unpartitioned_node_map,
    std::unordered_map<Node*, std::pair<string, string>>& nodes_to_add) {
  for (int k = 0; k < tensor_name_t.dim_size(0); ++k) {
    string tensor_n = tensor_name_t.flat<string>()(k);
    string s_and_s_s = shape_and_slice_t.flat<string>()(k);
    if (partitioned_variable_shape.find(tensor_n) ==
        partitioned_variable_shape.end()) {
      Node* n;
      TF_RETURN_IF_ERROR(
          cur_save_node->input_node(kSaverTensorInputOffset + k, &n));
      if (n->name() == "global_step/read") {
        continue;
      }
      if (n->name() == tensor_n) {
        nodes_to_add.emplace(n, std::pair<string, string>(tensor_n, s_and_s_s));
      } else if (n->type_string() == "Identity") {
        Node* identity_node;
        TF_RETURN_IF_ERROR(n->input_node(0, &identity_node));
        if (identity_node->type_string() == "Identity") {  // ResourceVariable
          Node* read_variable_node;
          TF_RETURN_IF_ERROR(identity_node->input_node(0, &read_variable_node));
          Node* resource_node;
          TF_RETURN_IF_ERROR(read_variable_node->input_node(0, &resource_node));
          if (unpartitioned_node_map.find(resource_node->name()) !=
              unpartitioned_node_map.end()) {
            nodes_to_add.emplace(
                n, std::pair<string, string>(tensor_n, s_and_s_s));
          }
        } else {  // RefVariable
          nodes_to_add.emplace(n,
                               std::pair<string, string>(tensor_n, s_and_s_s));
        }
      }
    }
  }
  return Status::OK();
}

Status DeleteOldSaverGraph(
    const std::vector<Node*>& save_node_vec,
    const std::unordered_map<string, std::vector<int64>>
        partitioned_variable_shape,
    const std::unordered_map<std::string, Node*>& unpartitioned_node_map, int i,
    int cur_partition_nums,
    std::unordered_map<Node*, std::pair<string, string>>& nodes_to_add,
    std::unordered_set<Node*>* nodes_to_delete) {
  Node* cur_save_node = save_node_vec[i];
  {
    Node* sharded_filename;
    TF_RETURN_IF_ERROR(cur_save_node->input_node(0, &sharded_filename));
    for (auto* o_node : sharded_filename->out_nodes()) {
      if (o_node->type_string() == "Identity") {
        nodes_to_delete->insert(o_node);
      } else if (o_node->type_string() == "Pack") {
        int part_num;
        TF_RETURN_IF_ERROR(GetNodeAttr(o_node->attrs(), "N", &part_num));
        if (part_num != cur_partition_nums) {
          o_node->ClearAttr("N");
          o_node->AddAttr("N", cur_partition_nums);
        }
      }
    }
    nodes_to_delete->insert(sharded_filename);
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
    TF_RETURN_IF_ERROR(FindReserverdNode(
        tensor_name_t, shape_and_slice_t, cur_save_node,
        partitioned_variable_shape, unpartitioned_node_map, nodes_to_add));
    nodes_to_delete->insert(tensor_names);
    nodes_to_delete->insert(shape_and_slices);
  }

  {
    Node* ev_names;
    TF_RETURN_IF_ERROR(cur_save_node->input_node(3, &ev_names));
    nodes_to_delete->insert(ev_names);
    Node* ev_resource;
    TF_RETURN_IF_ERROR(cur_save_node->input_node(4, &ev_resource));
    nodes_to_delete->insert(ev_resource);
  }
  nodes_to_delete->insert(cur_save_node);
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
