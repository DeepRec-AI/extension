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

#include "dynamic_embedding_server/include/graph/elastic_partition_pass.h"
#include "dynamic_embedding_server/include/utils/graph_utils.h"
#include "dynamic_embedding_server/include/utils/naming.h"
#include "tensorflow/core/common_runtime/function.h"

#include "include/json/json.h"

constexpr char kElasticImportScope[] = "elastic_import";
constexpr char kElasticSubGraphImport[] = "elastic_subgraph_import";
constexpr char kElasticSubGraphInit[] = "elastic_subgraph_init";
constexpr char kDatasetInit[] = "make_initializer";

namespace tensorflow {

int ElasticTrainingPass::cur_partition_num_ = 0;

Status UpdatePartitionNums(int& partition_nums) {
  std::string tf_config;
  ReadStringFromEnvVar("TF_CONFIG", "", &tf_config);

  Json::CharReaderBuilder builder;
  const std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
  Json::Value json_tf_config;
  string err;
  if (!reader->parse(tf_config.c_str(),
                     tf_config.c_str() + static_cast<int>(tf_config.length()),
                     &json_tf_config, &err)) {
    return Status(error::Code::INVALID_ARGUMENT, "PARSE TF_CONFIG ERROR");
  }

  if ((json_tf_config["cluster"].isNull()) ||
      (json_tf_config["cluster"]["ps"].isNull())) {
    return Status(error::Code::INVALID_ARGUMENT, "PARSE ps ERROR");
  }
  Json::Value ps_array = json_tf_config["cluster"]["ps"];
  partition_nums = ps_array.size();
  LOG(INFO) << " partition_nums is " << partition_nums;
  return Status::OK();
}

Status ElasticTrainingPass::Run(const GraphOptimizationPassOptions& options) {
  int partition_nums;
  TF_RETURN_IF_ERROR(UpdatePartitionNums(partition_nums));

  Graph* graph = options.graph->get();
  if (graph == nullptr) {
    return errors::Internal("a graph should be available.");
  }

  if (cur_partition_num_ == 0) {
    cur_partition_num_ = partition_nums;
    prev_partition_num_ = 1;
    return Status::OK();
  } else if (cur_partition_num_ == partition_nums) {
    LOG(INFO) << "No need to do elastic partition pass.";
    return Status::OK();
  } else {
    if (initialized_ == true) {
      prev_partition_num_ = cur_partition_num_;
    } else {
      prev_partition_num_ = 1;
      initialized_ = true;
    }
    delta_partition_num_ = partition_nums - cur_partition_num_;
    scaling_up_ = partition_nums > cur_partition_num_;
    cur_partition_num_ = partition_nums;
  }
  std::unique_ptr<Graph> new_graph(new Graph(OpRegistry::Global()));
  CopyGraph(*graph, new_graph.get());
  // VLogGraphDebugString(new_graph.get());
  TF_RETURN_IF_ERROR(RewriteSubGraph(new_graph.get()));
  DumpGraph("DynamicEmbeddingServer", new_graph.get());
  options.graph->swap(new_graph);
  // VLogGraphDebugString(options.graph->get());
  return Status::OK();
}

Status ElasticTrainingPass::RewriteSubGraph(Graph* g) {
  std::unordered_map<std::string, PartitionedVariable> primary_node_metas_map;
  std::unordered_map<std::string, Node*> unpartitioned_node_map;
  ElasticHookMetaNode meta_node(cur_partition_num_);
  TF_RETURN_IF_ERROR(meta_node.Init(g, prev_partition_num_));
  TF_RETURN_IF_ERROR(InitVarMeta(g, &meta_node, primary_node_metas_map,
                                 unpartitioned_node_map));
  TF_RETURN_IF_ERROR(
      RewriteTrainingSubGraph(g, primary_node_metas_map, &meta_node));
  TF_RETURN_IF_ERROR(RewriteSavingSubGraph(g, primary_node_metas_map,
                                           unpartitioned_node_map, meta_node));
  return Status::OK();
}

Status ElasticTrainingPass::InitVarMeta(
    Graph* g, ElasticHookMetaNode* meta_node,
    std::unordered_map<std::string, PartitionedVariable>&
        primary_node_metas_map,
    std::unordered_map<std::string, Node*>& unpartitioned_node_map) {
  for (auto* node : g->op_nodes()) {
    string node_name = node->name();
    int device_idx = 0;
    int partiton_size = 0;
    int partition_id = 0;
    VarType var_type;
    bool partitioned = false;
    if (node->IsKvVarHandle()) {
      var_type = VarType::EMBEDDING_VAR;
      TF_RETURN_IF_ERROR(FindNodeAndExec(
          node, ::des::kEvExportOp,
          [this, &device_idx, &partition_id](Node* target_node) {
            TF_RETURN_IF_ERROR(GetNodeAttr(target_node->attrs(), "partition_id",
                                           &partition_id));
            TF_RETURN_IF_ERROR(GetNodeAttr(target_node->attrs(), "device_id",
                                           &device_idx));
            return Status::OK();
          }));
      TF_RETURN_IF_ERROR(FindNodeAndExec(
          node, ::des::kEvImportOp, [this, &partitioned](Node* target_node) {
            int partition_num;
            TF_RETURN_IF_ERROR(GetNodeAttr(target_node->attrs(),
                                           "partition_nums", &partition_num));
            partitioned = partition_num > 1;
            return Status::OK();
          }));
    } else if (node->IsVariable()) {
      if (IsRefType(node->output_type(0))) {
        var_type = VarType::DENSE_REF_VAR;
        TensorShape tensor_shape;
        TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "shape", &tensor_shape));
        partiton_size = tensor_shape.dim_size(0);
        TF_RETURN_IF_ERROR(FindNodeAndExec(
            node, kIdentityOp, [this, &var_type](Node* target_node) {
              for (auto* oo_node : target_node->out_nodes()) {
                if (oo_node->type_string() == "GatherV2") {
                  var_type = VarType::REF_VAR;
                }
              }
              return Status::OK();
            }));
        TF_RETURN_IF_ERROR(FindNodeAndExec(
            node, ::des::kReAssign,
            [this, &device_idx, &partitioned,
             &partition_id](Node* target_node) {
              TF_RETURN_IF_ERROR(
                  GetNodeAttr(target_node->attrs(), "device_id", &device_idx));
              TF_RETURN_IF_ERROR(GetNodeAttr(target_node->attrs(),
                                             "partition_id", &partition_id));
              partitioned = true;
              return Status::OK();
            }));
      }
    } else if (node->type_string() == "VarHandleOp") {
      var_type = VarType::DENSE_RESOUCE_VAR;
      TensorShape tensor_shape;
      TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "shape", &tensor_shape));
      partiton_size = tensor_shape.dim_size(0);
      TF_RETURN_IF_ERROR(FindNodeAndExec(node, "ResourceGather",
                                         [this, &var_type](Node* target_node) {
                                           var_type = VarType::RESOURCE_VAR;
                                           return Status::OK();
                                         }));
      TF_RETURN_IF_ERROR(FindNodeAndExec(
          node, ::des::kReAssignRes,
          [this, &device_idx, &partitioned, &partition_id](Node* target_node) {
            TF_RETURN_IF_ERROR(
                GetNodeAttr(target_node->attrs(), "device_id", &device_idx));
            TF_RETURN_IF_ERROR(GetNodeAttr(target_node->attrs(), "partition_id",
                                           &partition_id));
            partitioned = true;
            return Status::OK();
          }));
    } else {
      continue;
    }
    // Track the partitioned variable
    if (partitioned) {
      auto part_idx = node_name.find(kPart);
      std::string pre_str = node_name.substr(0, part_idx - 1);
      std::string post_str = node_name.substr(part_idx + strlen(kPart));
      auto post_idx = post_str.find("/");
      if (post_idx == string::npos) {
        if (primary_node_metas_map.find(pre_str) ==
            primary_node_metas_map.end()) {
          PartitionedVariable partition_var(g, meta_node);
          partition_var.variable_prefix = pre_str;
          partition_var.var_type = var_type;
          partition_var.part_var_full_shape = partiton_size;
          partition_var.ev_partition_num = 1;
          partition_var.node_map =
              std::unordered_map<std::string, PartIdToNodeMap>{
                  {pre_str, PartIdToNodeMap{{partition_id, node}}}};
          partition_var.node_device_map =
              std::unordered_map<Node*, int>{{node, device_idx}};
          // exactly once
          partition_var.opt_ev_names = std::unordered_set<string>{};
          primary_node_metas_map.emplace(pre_str, std::move(partition_var));
        } else {
          primary_node_metas_map[pre_str].part_var_full_shape += partiton_size;
          primary_node_metas_map[pre_str].ev_partition_num++;
          primary_node_metas_map[pre_str].node_map[pre_str].emplace(
              DeviceIdToNodePair(partition_id, node));
          primary_node_metas_map[pre_str].node_device_map.emplace(node,
                                                                  device_idx);
        }
      } else {
        string opt_name = pre_str + post_str.substr(post_idx);
        if (primary_node_metas_map.find(pre_str) ==
            primary_node_metas_map.end()) {
          PartitionedVariable partition_var(g, meta_node);
          partition_var.variable_prefix = pre_str;
          partition_var.var_type = var_type;
          partition_var.part_var_full_shape = partiton_size;
          partition_var.ev_partition_num = 1;
          partition_var.node_map =
              std::unordered_map<std::string, PartIdToNodeMap>{
                  {opt_name, PartIdToNodeMap{{partition_id, node}}}};
          partition_var.node_device_map =
              std::unordered_map<Node*, int>{{node, device_idx}};
          partition_var.opt_ev_names = std::unordered_set<string>{opt_name};
          primary_node_metas_map.emplace(pre_str, std::move(partition_var));
        } else {
          primary_node_metas_map[pre_str].node_map[opt_name].emplace(
              DeviceIdToNodePair(partition_id, node));
          primary_node_metas_map[pre_str].opt_ev_names.insert(opt_name);
          primary_node_metas_map[pre_str].node_device_map.emplace(node,
                                                                  device_idx);
        }
      }
    } else {
      if (node_name.find(kElasticImportScope) == string::npos) {
        unpartitioned_node_map.emplace(node_name, node);
      }
    }
  }
  return Status::OK();
}

Status ElasticTrainingPass::RewriteTrainingSubGraph(
    Graph* g,
    std::unordered_map<std::string, PartitionedVariable>&
        primary_node_metas_map,
    ElasticHookMetaNode* meta_node) {
  for (auto& it : primary_node_metas_map) {
    auto& partition_var = it.second;
    partition_var.Prepare(cur_partition_num_, prev_partition_num_,
                          delta_partition_num_);
    TF_RETURN_IF_ERROR(partition_var.Scaling(prev_partition_num_));
  }
  return Status::OK();
}

Status ElasticTrainingPass::RewriteSavingSubGraph(
    Graph* g,
    std::unordered_map<std::string, PartitionedVariable>&
        primary_node_metas_map,
    std::unordered_map<std::string, Node*>& unpartitioned_node_map,
    ElasticHookMetaNode& meta_node) {
  std::vector<Node*> save_node_vec;
  std::vector<Node*> restore_node_vec;
  for (auto* node : g->nodes()) {
    if (node->type_string() == kSaveOp) {
      Node* sharded_filename;
      TF_RETURN_IF_ERROR(node->input_node(0, &sharded_filename));
      if (sharded_filename->type_string() != "Const") {
        save_node_vec.emplace_back(node);
      }
    } else if (node->type_string() == kRestoreOp) {
      Node* sharded_filename;
      TF_RETURN_IF_ERROR(node->input_node(0, &sharded_filename));
      if (sharded_filename->type_string() != "Const") {
        restore_node_vec.emplace_back(node);
      }
    }
  }

  if ((save_node_vec.size() == 0) || (restore_node_vec.size() == 0)) {
    LOG(INFO) << "There is no Save and Restore Op in Graph, Nothing to do.";
    return Status::OK();
  }

  std::unordered_set<Node*> nodes_to_delete;
  std::unordered_map<string, std::vector<int64>> partitioned_variable_shape;
  std::unordered_map<string, string> opt_to_primary_map;
  std::unordered_map<Node*, std::pair<string, string>> nodes_to_add;
  std::unordered_set<Node*> eval_nodes_to_add;
  Node* pack_node = nullptr;
  int axis = 0;
  DataType concat_type;
  std::vector<Node*> pack_inputs;

  InitPartVariableShape(primary_node_metas_map, partitioned_variable_shape,
                        opt_to_primary_map);

  if (scaling_up_) {
    for (int i = 0; i < cur_partition_num_; ++i) {
      if (i < save_node_vec.size()) {
        TF_RETURN_IF_ERROR(RewritePrevSaveGraph(
            g, i, cur_partition_num_, scaling_up_, save_node_vec, pack_inputs,
            concat_type, primary_node_metas_map, unpartitioned_node_map,
            partitioned_variable_shape, nodes_to_add, &nodes_to_delete,
            eval_nodes_to_add, opt_to_primary_map));
      } else {
        Status s;
        std::vector<Node*> restore_tensor_vec;
        std::vector<string> tensor_names_vec;
        std::vector<DataType> n_dtypes;
        bool has_ev = false;
        std::string assigned_device_name =
            "/job:ps/replica:0/task:" + std::to_string(i) + "/device:CPU:0";
        Node* new_sharded_filename;
        TF_RETURN_IF_ERROR(AddNewSaverGraph(
            g, has_ev, &new_sharded_filename, tensor_names_vec, n_dtypes,
            pack_inputs, axis, &pack_node, primary_node_metas_map,
            partitioned_variable_shape, restore_tensor_vec,
            assigned_device_name, save_node_vec, i, cur_partition_num_));
        TF_RETURN_IF_ERROR(AddNewRestoreGraph(
            g, has_ev, i, cur_partition_num_, new_sharded_filename,
            tensor_names_vec, partitioned_variable_shape, restore_tensor_vec,
            restore_node_vec, assigned_device_name, n_dtypes));
      }
    }
    TF_RETURN_IF_ERROR(MakePackOp(g, pack_inputs, pack_node, concat_type, axis,
                                  cur_partition_num_));
  } else if (save_node_vec.size() > cur_partition_num_) {
    for (int i = save_node_vec.size() - 1; i >= 0; --i) {
      if (i >= cur_partition_num_) {
        TF_RETURN_IF_ERROR(DeleteOldSaverGraph(
            save_node_vec, partitioned_variable_shape, unpartitioned_node_map,
            i, cur_partition_num_, nodes_to_add, &nodes_to_delete));
      } else {
        TF_RETURN_IF_ERROR(RewritePrevSaveGraph(
            g, i, cur_partition_num_, scaling_up_, save_node_vec, pack_inputs,
            concat_type, primary_node_metas_map, unpartitioned_node_map,
            partitioned_variable_shape, nodes_to_add, &nodes_to_delete,
            eval_nodes_to_add, opt_to_primary_map));
      }
    }

    TF_RETURN_IF_ERROR(ProcessUnPartitionedVariable(
        g, meta_node, nodes_to_add, eval_nodes_to_add, unpartitioned_node_map));
  }

  for (auto* n : nodes_to_delete) {
    g->RemoveNode(n);
  }

  return Status::OK();
}

Status ElasticHookMetaNode::Init(Graph* g, int prev_partition_num) {
  Node* dataset_init = nullptr;
  for (auto* node : g->nodes()) {
    if (node->name() == kElasticSubGraphImport) {
      m_import_op_main = node;
    } else if (node->name() == kElasticSubGraphInit) {
      m_init_op_main = node;
    } else if (node->name() == kDatasetInit) {
      dataset_init = node;
    }
  }

  if ((dataset_init != nullptr) && (m_init_op_main != nullptr)) {
    g->AddControlEdge(dataset_init, m_init_op_main);
  }

  Status s;
  for (int i = 0; i < m_num_partition; ++i) {
    string new_device_name =
        "/job:ps/replica:0/task:" + std::to_string(i) + "/device:CPU:0";
    NodeDef initop_def;
    TF_RETURN_IF_ERROR(
        NodeDefBuilder("new_sub_graph/InitOp_" + std::to_string(i), "NoOp")
            .Device(new_device_name)
            .Finalize(&initop_def));
    Node* init_node = g->AddNode(initop_def, &s);
    init_node->set_assigned_device_name(new_device_name);
    TF_RETURN_IF_ERROR(s);
    m_init_op_vec[i] = init_node;
    if (i >= prev_partition_num) {
      g->AddControlEdge(m_init_op_vec[i], m_init_op_main);
    }

    bool found = false;
    for (auto* node : g->op_nodes()) {
      if (node->name() == "head/Optimizer/update/NoOp_" + std::to_string(i)) {
        m_no_op_vec[i] = node;
        found = true;
        break;
      }
    }
    if (!found) {
      NodeDef noop_def;
      TF_RETURN_IF_ERROR(
          NodeDefBuilder("head/Optimizer/update/NoOp_" + std::to_string(i),
                         "NoOp")
              .Device(new_device_name)
              .Finalize(&noop_def));
      Node* no_node = g->AddNode(noop_def, &s);
      no_node->set_assigned_device_name(new_device_name);
      m_no_op_vec[i] = no_node;
      TF_RETURN_IF_ERROR(s);
    }
  }

  NodeDef initop_def;
  TF_RETURN_IF_ERROR(NodeDefBuilder("new_sub_graph/tmp_value/InitOp", "NoOp")
                         .Device("/job:worker/replica:0/task:0/device:CPU:0")
                         .Finalize(&initop_def));
  m_tmp_value_init_op = g->AddNode(initop_def, &s);
  m_tmp_value_init_op->set_assigned_device_name(
      "/job:worker/replica:0/task:0/device:CPU:0");
  TF_RETURN_IF_ERROR(s);
  g->AddControlEdge(m_tmp_value_init_op, m_init_op_main);
  return s;
}

Status PartitionedVariable::ScalingDownEVForWardGraph(
    std::vector<bool>& part_to_scale_down,
    std::unordered_set<Node*>* nodes_to_delete) {
  for (int i = 0; i < part_to_scale_down.size(); ++i) {
    if (part_to_scale_down[i]) {
      Node* var_node = node_map[variable_prefix][i];
      for (auto* o_node : var_node->out_nodes()) {
        // InitializeEVResource
        if (o_node->type_string() == kEvInitOp) {
          const Node* tmp_check_ev_0;
          TF_RETURN_IF_ERROR(o_node->input_node(0, &tmp_check_ev_0));
          const Node* tmp_check_ev_1;
          TF_RETURN_IF_ERROR(o_node->input_node(1, &tmp_check_ev_1));
          if (tmp_check_ev_0->name() != tmp_check_ev_1->name()) {
            continue;
          }

          nodes_to_delete->insert(o_node);
          const Edge* init_value_edge = nullptr;
          TF_RETURN_IF_ERROR(o_node->input_edge(2, &init_value_edge));
          nodes_to_delete->insert(init_value_edge->src());
          const Edge* empty_key_edge = nullptr;
          TF_RETURN_IF_ERROR(o_node->input_edge(3, &empty_key_edge));
          nodes_to_delete->insert(empty_key_edge->src());
        } else if (o_node->type_string() == "KvResourceGather") {
          nodes_to_delete->insert(o_node);
          for (auto* o_edge : o_node->out_edges()) {
            if (o_edge->dst()->type_string() == kIdentityOp) {
              nodes_to_delete->insert(o_edge->dst());
            }
          }
        }
      }

      for (auto& opt_ev_name : sorted_opt_ev_names) {
        Node* ev_node = node_map[opt_ev_name][i];
        for (auto* o_node : ev_node->out_nodes()) {
          if (o_node->type_string() == kEvInitOp) {
            nodes_to_delete->insert(o_node);
            const Edge* init_value_edge = nullptr;
            TF_RETURN_IF_ERROR(o_node->input_edge(2, &init_value_edge));
            nodes_to_delete->insert(init_value_edge->src());
            const Edge* empty_key_edge = nullptr;
            TF_RETURN_IF_ERROR(o_node->input_edge(3, &empty_key_edge));
            nodes_to_delete->insert(empty_key_edge->src());
          }
        }
      }
    }
  }
  return Status::OK();
}

Status PartitionedVariable::ScalingUpEVForWardGraph(
    int original_indices, std::vector<int>& part_to_device) {
  // This function is responsible for Create new EV handler and its
  // init subgraph. For primary EV, we also create Gather graph.
  for (int i = ev_partition_num; i < cur_partition_num; ++i) {
    Node* cur_init_op = meta_node->m_init_op_vec[i];
    Node* ori_ev_node = node_map[variable_prefix][original_indices];
    int device_id = part_to_device[i];
    string new_device_name = NewDevice(ori_ev_node, device_id);
    std::string op_name = variable_prefix + "/" + kPart + std::to_string(i);

    Node* new_ev_node = CopyNode(g, ori_ev_node, new_device_name, i, op_name);
    new_ev_node->ClearAttr("shared_name");
    new_ev_node->AddAttr("shared_name", op_name);
    new_ev_node->AddAttr("_class", "loc:@" + op_name);
    node_map[variable_prefix].emplace(i, new_ev_node);

    bool is_init = false;
    Node* primary_init_node;
    TF_RETURN_IF_ERROR(MakeEVInitializer(ori_ev_node, new_ev_node, is_init, i,
                                         g, cur_init_op, &primary_init_node,
                                         new_device_name));

    TF_RETURN_IF_ERROR(
        MakeResourceGatherOp(g, ori_ev_node, new_ev_node, new_device_name, i));

    // OptEV
    for (auto& opt_ev_name : sorted_opt_ev_names) {
      auto opt_var_node = node_map[opt_ev_name][original_indices];
      auto sep_idx = opt_ev_name.rfind("/");
      std::string op_name = opt_ev_name.substr(0, sep_idx) + "/" + kPart +
                            std::to_string(i) + opt_ev_name.substr(sep_idx);

      // EVHandler
      Node* new_opt_ev_node =
          CopyNode(g, opt_var_node, new_device_name, i, op_name);
      new_opt_ev_node->ClearAttr("shared_name");
      new_opt_ev_node->AddAttr("shared_name", op_name);
      node_map[opt_ev_name].emplace(i, new_opt_ev_node);

      is_init = false;
      TF_RETURN_IF_ERROR(MakeEVInitializer(
          opt_var_node, new_ev_node, is_init, i, g, cur_init_op,
          &primary_init_node, new_device_name, new_opt_ev_node));
    }
  }
  return Status::OK();
}

Status PartitionedVariable::ScalingDownResVarForWardGraph(
    PartIdToNodeMap& node_vec, std::vector<bool>& part_to_scale_down,
    std::unordered_set<Node*>* nodes_to_delete) {
  std::unordered_set<Node*> concat_inputs;
  std::unordered_set<Node*> elastic_concat_inputs;
  Node* concat_node = nullptr;
  Node* elastic_concat_node = nullptr;
  for (int i = 0; i < part_to_scale_down.size(); ++i) {
    Node* ev_node = node_vec[i];
    if (!part_to_scale_down[i]) {
      TensorShape new_shape;
      TF_RETURN_IF_ERROR(ChangeResShape(ev_node, new_shape, part_var_full_shape,
                                        cur_partition_num, false));

      for (auto* o_node : ev_node->out_nodes()) {
        if (o_node->type_string() == "ReadVariableOp") {
          for (auto* o_edge : o_node->out_edges()) {
            // Normal Variable
            if (o_edge->dst()->type_string() == "ConcatV2") {
              if (o_edge->dst()->name().find(kElasticImportScope) ==
                  string::npos) {
                int N;
                TF_RETURN_IF_ERROR(
                    GetNodeAttr(o_edge->dst()->attrs(), "N", &N));
                if (N != cur_partition_num) {
                  const Edge* axis_edge = nullptr;
                  TF_RETURN_IF_ERROR(o_edge->dst()->input_edge(N, &axis_edge));
                  concat_inputs.insert(axis_edge->src());
                  o_edge->dst()->ClearAttr("N");
                  o_edge->dst()->AddAttr("N", cur_partition_num);
                }
                concat_node = o_edge->dst();
                concat_inputs.insert(o_node);
              } else {
                // exactly once
                int N;
                TF_RETURN_IF_ERROR(
                    GetNodeAttr(o_edge->dst()->attrs(), "N", &N));
                if (N != cur_partition_num) {
                  const Edge* axis_edge;
                  TF_RETURN_IF_ERROR(o_edge->dst()->input_edge(N, &axis_edge));
                  elastic_concat_inputs.insert(axis_edge->src());
                  o_edge->dst()->ClearAttr("N");
                  o_edge->dst()->AddAttr("N", cur_partition_num);
                }
                elastic_concat_node = o_edge->dst();
                elastic_concat_inputs.insert(o_node);
              }
            }
          }
        }
      }
    } else {
      for (auto* o_node : ev_node->out_nodes()) {
        if (o_node->type_string() == "ResourceGather") {
          nodes_to_delete->insert(o_node);
          for (auto* oo_node : o_node->out_nodes()) {
            if (oo_node->type_string() == kIdentityOp) {
              nodes_to_delete->insert(oo_node);
            }
          }
        } else if (o_node->type_string() == "AssignVariableOp") {
          nodes_to_delete->insert(o_node);
        }
      }
    }
  }

  if (concat_node != nullptr) {
    std::vector<Node*> input_node;
    std::vector<const Edge*> edges_to_delete;
    for (auto* edge : concat_node->in_edges()) {
      if (concat_inputs.find(edge->src()) != concat_inputs.end()) {
        input_node.emplace_back(edge->src());
      }
      edges_to_delete.emplace_back(edge);
    }
    for (auto* edge : edges_to_delete) {
      g->RemoveEdge(edge);
    }
    for (int i = 0; i < input_node.size(); ++i) {
      g->AddEdge(input_node[i], 0, concat_node, i);
    }
  }
  if (elastic_concat_node != nullptr) {
    std::vector<Node*> input_node;
    std::vector<const Edge*> edges_to_delete;
    for (auto* edge : elastic_concat_node->in_edges()) {
      if (elastic_concat_inputs.find(edge->src()) !=
          elastic_concat_inputs.end()) {
        input_node.emplace_back(edge->src());
      }
      edges_to_delete.emplace_back(edge);
    }
    for (auto* edge : edges_to_delete) {
      g->RemoveEdge(edge);
    }
    for (int i = 0; i < input_node.size(); ++i) {
      g->AddEdge(input_node[i], 0, elastic_concat_node, i);
    }
  }
  return Status::OK();
}

Status PartitionedVariable::ScalingUpResVarForWardGraph(
    int original_indices, std::vector<int>& part_to_device,
    std::unordered_set<Node*>* nodes_to_delete) {
  auto& var_vec = node_map[variable_prefix];
  for (int i = 0; i < cur_partition_num; ++i) {
    if (i < ev_partition_num) {
      Node* var_node = var_vec[i];
      TensorShape var_shape;
      TF_RETURN_IF_ERROR(ChangeResShape(
          var_node, var_shape, part_var_full_shape, cur_partition_num, false));

      for (auto& opt_ev_name : sorted_opt_ev_names) {
        auto opt_var_node = node_map[opt_ev_name][i];
        TF_RETURN_IF_ERROR(ChangeResShape(opt_var_node, var_shape,
                                          part_var_full_shape,
                                          cur_partition_num, false));
      }
    } else {
      Node* var_node = var_vec[original_indices];
      Node* cur_init_op = meta_node->m_init_op_vec[i];
      int device_id = part_to_device[i];
      string new_device_name = NewDevice(var_node, device_id);
      std::string op_name = variable_prefix + "/" + kPart + std::to_string(i);
      Node* new_var_node = CopyNode(g, var_node, new_device_name, i, op_name);
      TensorShape new_shape;
      TF_RETURN_IF_ERROR(ChangeResShape(new_var_node, new_shape,
                                        part_var_full_shape, cur_partition_num,
                                        i == cur_partition_num - 1));

      var_vec.emplace(i, new_var_node);

      TF_RETURN_IF_ERROR(FindNodeAndExec(
          var_node, "AssignVariableOp",
          [this, &new_var_node, &cur_init_op, new_shape, new_device_name,
           i](Node* target_node) {
            Status s;
            if (target_node->name().find("save") == string::npos) {
              Node* new_var_init = CopyNode(g, target_node, new_device_name, i);
              g->AddEdge(new_var_node, 0, new_var_init, 0);
              TF_RETURN_IF_ERROR(MakeConstInitializer(
                  g, new_var_init, new_shape, new_device_name));
              g->AddControlEdge(new_var_init, cur_init_op);
            }
            return s;
          }));

      TF_RETURN_IF_ERROR(FindNodeAndExec(
          var_node, "ResourceGather",
          [this, &new_var_node, new_device_name, i](Node* target_node) {
            Node* gather_op = CopyNode(g, target_node, new_device_name, i);
            g->AddEdge(new_var_node, 0, gather_op, 0);
            const Edge* gather_id_edge = nullptr;
            TF_RETURN_IF_ERROR(target_node->input_edge(1, &gather_id_edge));
            g->AddEdge(gather_id_edge->src(), gather_id_edge->src_output(),
                       gather_op, 1);
            for (auto* o_edge : target_node->out_edges()) {
              if (o_edge->dst()->type_string() == kIdentityOp) {
                Node* identity_op =
                    CopyNode(g, o_edge->dst(), new_device_name, i);
                g->AddEdge(gather_op, 0, identity_op, 0);
              }
            }
            return Status::OK();
          }));

      TF_RETURN_IF_ERROR(FindNodeAndExec(
          var_node, "ReadVariableOp",
          [this, &new_var_node, new_device_name, i](Node* target_node) {
            Node* new_var_read = CopyNode(g, target_node, new_device_name, i);
            g->AddEdge(new_var_node, 0, new_var_read, 0);
            for (auto* oo_node : target_node->out_nodes()) {
              // Normal Variable
              if ((oo_node->type_string() == "ConcatV2") &&
                  (oo_node->name().find(kElasticImportScope) == string::npos)) {
                int N;
                TF_RETURN_IF_ERROR(GetNodeAttr(oo_node->attrs(), "N", &N));
                if (N != cur_partition_num) {
                  const Edge* axis_edge;
                  TF_RETURN_IF_ERROR(oo_node->input_edge(N, &axis_edge));
                  oo_node->ClearAttr("N");
                  oo_node->AddAttr("N", cur_partition_num);
                  g->RemoveEdge(axis_edge);
                  g->AddEdge(axis_edge->src(), 0, oo_node, cur_partition_num);
                }
                g->AddEdge(new_var_read, 0, oo_node, i);
              }
            }
            return Status::OK();
          }));

      for (auto& opt_ev_name : sorted_opt_ev_names) {
        auto var_node = node_map[opt_ev_name][original_indices];
        auto sep_idx = opt_ev_name.rfind("/");
        std::string op_name = opt_ev_name.substr(0, sep_idx) + "/" + kPart +
                              std::to_string(i) + opt_ev_name.substr(sep_idx);

        Node* new_opt_var_node =
            CopyNode(g, var_node, new_device_name, i, op_name);
        TensorShape new_shape;
        TF_RETURN_IF_ERROR(
            ChangeResShape(new_opt_var_node, new_shape, part_var_full_shape,
                           cur_partition_num, i == cur_partition_num - 1));

        node_map[opt_ev_name].emplace(i, new_opt_var_node);

        TF_RETURN_IF_ERROR(FindNodeAndExec(
            var_node, "AssignVariableOp",
            [this, &new_opt_var_node, &cur_init_op, new_shape, new_device_name,
             i](Node* target_node) {
              if (target_node->name().find("save") == string::npos) {
                Node* new_var_init =
                    CopyNode(g, target_node, new_device_name, i);
                g->AddEdge(new_opt_var_node, 0, new_var_init, 0);
                TF_RETURN_IF_ERROR(MakeConstInitializer(
                    g, new_var_init, new_shape, new_device_name));
                g->AddControlEdge(new_var_init, cur_init_op);
              }
              return Status::OK();
            }));

        TF_RETURN_IF_ERROR(FindNodeAndExec(
            var_node, "ReadVariableOp",
            [this, &new_opt_var_node, new_device_name, i](Node* target_node) {
              Node* new_var_read = CopyNode(g, target_node, new_device_name, i);
              g->AddEdge(new_opt_var_node, 0, new_var_read, 0);
              for (auto* oo_node : target_node->out_nodes()) {
                if (oo_node->type_string() == "Identity") {
                  for (auto* ooo_node : oo_node->out_nodes()) {
                    if (ooo_node->type_string() == "Identity") {
                      Node* identity_read =
                          CopyNode(g, oo_node, new_device_name, i);
                      g->AddEdge(new_var_read, 0, identity_read, 0);
                      Node* new_identity_read =
                          CopyNode(g, ooo_node, new_device_name, i);
                      g->AddEdge(identity_read, 0, new_identity_read, 0);
                    }
                  }
                }
                // Normal Variable
                if ((oo_node->type_string() == "ConcatV2") &&
                    (oo_node->name().find(kElasticImportScope) ==
                     string::npos)) {
                  int N;
                  TF_RETURN_IF_ERROR(GetNodeAttr(oo_node->attrs(), "N", &N));
                  if (N != cur_partition_num) {
                    const Edge* axis_edge;
                    TF_RETURN_IF_ERROR(oo_node->input_edge(N, &axis_edge));
                    oo_node->ClearAttr("N");
                    oo_node->AddAttr("N", cur_partition_num);
                    g->RemoveEdge(axis_edge);
                    g->AddEdge(axis_edge->src(), 0, oo_node, cur_partition_num);
                  }
                  g->AddEdge(new_var_read, 0, oo_node, i);
                }
              }
              return Status::OK();
            }));
      }
    }
  }
  return Status::OK();
}

Status PartitionedVariable::ScalingDownVarForWardGraph(
    PartIdToNodeMap& node_vec, std::vector<bool>& part_to_scale_down,
    std::unordered_set<Node*>* nodes_to_delete) {
  Node* concat_node = nullptr;
  Node* axis_node = nullptr;
  Node* elastic_concat_node = nullptr;
  Node* elastic_axis_node = nullptr;
  DataType concat_type;
  std::vector<Node*> concat_inputs;
  std::vector<Node*> elastic_concat_inputs;

  for (int i = 0; i < part_to_scale_down.size(); ++i) {
    Node* tmp_var_node = node_vec[i];
    if (!part_to_scale_down[i]) {
      TF_RETURN_IF_ERROR(FindNodeAndExec(
          tmp_var_node, kIdentityOp,
          [this, &concat_inputs, &concat_node, &elastic_concat_inputs,
           &concat_type, &elastic_concat_node, &axis_node, &elastic_axis_node,
           i](Node* target_node) {
            for (auto* o_node : target_node->out_nodes()) {
              // Normal Variable
              if (o_node->type_string() == "ConcatV2") {
                if (o_node->name().find(kElasticImportScope) == string::npos) {
                  concat_node = o_node;
                  concat_inputs.emplace_back(target_node);
                  int N;
                  TF_RETURN_IF_ERROR(GetNodeAttr(o_node->attrs(), "N", &N));
                  TF_RETURN_IF_ERROR(o_node->input_node(N, &axis_node));
                  TF_RETURN_IF_ERROR(
                      GetNodeAttr(o_node->attrs(), "T", &concat_type));
                } else {
                  elastic_concat_node = o_node;
                  elastic_concat_inputs.emplace_back(target_node);
                  int N;
                  TF_RETURN_IF_ERROR(GetNodeAttr(o_node->attrs(), "N", &N));
                  TF_RETURN_IF_ERROR(o_node->input_node(N, &elastic_axis_node));
                }
              }
            }
            return Status::OK();
          }));
    } else {
      for (auto* o_node : tmp_var_node->out_nodes()) {
        if (o_node->type_string() == kIdentityOp) {
          nodes_to_delete->insert(o_node);
          for (auto* oo_node : o_node->out_nodes()) {
            if (oo_node->type_string() == "GatherV2") {
              nodes_to_delete->insert(oo_node);
            }
          }
        } else if (o_node->type_string() == "Assign") {
          nodes_to_delete->insert(o_node);
        }
      }
    }
  }

  TF_RETURN_IF_ERROR(MakeConcatOp(g, concat_node, concat_inputs, axis_node,
                                  concat_type, cur_partition_num));
  TF_RETURN_IF_ERROR(MakeConcatOp(g, elastic_concat_node, elastic_concat_inputs,
                                  elastic_axis_node, concat_type,
                                  cur_partition_num));
  return Status::OK();
}

Status PartitionedVariable::ScalingUpVarForWardGraph(
    int original_indices, const std::string& var_name,
    std::vector<int>& part_to_device, int prev_partition_num) {
  auto& node_vec = node_map[var_name];
  Node* var_node = node_vec[original_indices];
  Node* elastic_concat_node = nullptr;
  Node* elastic_axis_node = nullptr;
  DataType concat_type;
  std::vector<Node*> elastic_concat_inputs;
  std::unordered_map<std::string, ConcatMeta> concat_metas;
  for (int i = 0; i < cur_partition_num; ++i) {
    if (i < ev_partition_num) {
      Node* tmp_var_node = node_vec[i];
      TF_RETURN_IF_ERROR(ChangeVarShape(tmp_var_node, part_var_full_shape,
                                        cur_partition_num, i, "shape"));
      TF_RETURN_IF_ERROR(
          FindNodeAndExec(tmp_var_node, "Assign", [this, i](Node* target_node) {
            if (target_node->name().find("save") == -1) {
              const Edge* init_value_edge = nullptr;
              TF_RETURN_IF_ERROR(target_node->input_edge(1, &init_value_edge));
              TF_RETURN_IF_ERROR(
                  ChangeVarShape(init_value_edge->src(), part_var_full_shape,
                                 cur_partition_num, i, "init_value"));
            }
            return Status::OK();
          }));

      TF_RETURN_IF_ERROR(FindNodeAndExec(
          tmp_var_node, kIdentityOp,
          [this, &concat_metas, &elastic_concat_inputs, &elastic_concat_node,
           &elastic_axis_node, &concat_type, i,
           prev_partition_num](Node* target_node) {
            for (auto* o_node : target_node->out_nodes()) {
              // Normal Variable
              if (o_node->type_string() == "ConcatV2") {
                if (o_node->name().find(kElasticImportScope) != string::npos) {
                  // when scaling multiple times, the actual variable nums is
                  // prev_ps_num
                  if ((prev_partition_num > 1) && (i < prev_partition_num)) {
                    // exactly once
                    elastic_concat_inputs.emplace_back(target_node);
                    elastic_concat_node = o_node;
                    int N;
                    TF_RETURN_IF_ERROR(GetNodeAttr(o_node->attrs(), "N", &N));
                    TF_RETURN_IF_ERROR(
                        o_node->input_node(N, &elastic_axis_node));
                    TF_RETURN_IF_ERROR(
                        GetNodeAttr(o_node->attrs(), "T", &concat_type));
                  }
                } else {
                  auto it = concat_metas.find(o_node->name());
                  if (it == concat_metas.end()) {
                    concat_metas.emplace(o_node->name(), ConcatMeta());
                  }
                  auto& concat_meta = concat_metas[o_node->name()];
                  concat_meta.concat_node = o_node;
                  concat_meta.concat_inputs.emplace_back(target_node);
                  int N;
                  TF_RETURN_IF_ERROR(GetNodeAttr(o_node->attrs(), "N", &N));
                  TF_RETURN_IF_ERROR(
                      o_node->input_node(N, &concat_meta.axis_node));
                  TF_RETURN_IF_ERROR(
                      GetNodeAttr(o_node->attrs(), "T", &concat_type));
                }
              }
            }
            return Status::OK();
          }));
    } else {
      Node* cur_init_op = meta_node->m_init_op_vec[i];
      int device_id = part_to_device[i];
      string new_device_name = NewDevice(var_node, device_id);
      std::string op_name = var_name + "/" + kPart + std::to_string(i);
      Node* new_var_node = CopyNode(g, var_node, new_device_name, i, op_name);
      TF_RETURN_IF_ERROR(ChangeVarShape(new_var_node, part_var_full_shape,
                                        cur_partition_num, i, "shape"));
      node_vec.emplace(i, new_var_node);
      node_device_map.emplace(new_var_node, device_id);
      TF_RETURN_IF_ERROR(FindNodeAndExec(
          var_node, "Assign",
          [this, &new_var_node, &cur_init_op, new_device_name,
           i](Node* target_node) {
            if (target_node->name().find("save") == -1) {
              Node* new_var_init = CopyNode(g, target_node, new_device_name, i);
              g->AddEdge(new_var_node, 0, new_var_init, 0);

              const Edge* init_value_edge = nullptr;
              TF_RETURN_IF_ERROR(target_node->input_edge(1, &init_value_edge));
              Node* new_init_value =
                  CopyNode(g, init_value_edge->src(), new_device_name, i);
              TF_RETURN_IF_ERROR(
                  ChangeVarShape(new_init_value, part_var_full_shape,
                                 cur_partition_num, i, "init_value"));

              g->AddEdge(new_init_value, 0, new_var_init, 1);
              g->AddControlEdge(new_var_init, cur_init_op);
            }
            return Status::OK();
          }));

      TF_RETURN_IF_ERROR(FindNodeAndExec(
          var_node, kIdentityOp,
          [this, &new_var_node, &concat_metas, &elastic_concat_inputs,
           &elastic_concat_node, &elastic_axis_node, new_device_name, i,
           prev_partition_num](Node* target_node) {
            Node* new_var_read = CopyNode(g, target_node, new_device_name, i);
            g->AddEdge(new_var_node, 0, new_var_read, 0);
            for (auto* oo_node : target_node->out_nodes()) {
              // Normal Variable
              if (oo_node->type_string() == "ConcatV2") {
                if (oo_node->name().find(kElasticImportScope) != string::npos) {
                  if ((prev_partition_num > 1) && (i < prev_partition_num)) {
                    elastic_concat_inputs.emplace_back(new_var_read);
                  }
                } else {
                  auto it = concat_metas.find(oo_node->name());
                  if (it == concat_metas.end()) {
                    concat_metas.emplace(oo_node->name(), ConcatMeta());
                  }
                  auto& concat_meta = concat_metas[oo_node->name()];
                  concat_meta.concat_inputs.emplace_back(new_var_read);
                }
              } else if (oo_node->type_string() == "GatherV2") {
                std::string new_gather_name =
                    oo_node->name().substr(0,
                                           oo_node->name().find("GatherV2")) +
                    "GatherV2_" + std::to_string(i);
                Node* new_gather =
                    CopyNode(g, oo_node, new_device_name, i, new_gather_name);

                g->AddEdge(new_var_read, 0, new_gather, 0);
                Node* axis_node = nullptr;
                TF_RETURN_IF_ERROR(oo_node->input_node(2, &axis_node));
                Node* new_axis_node =
                    CopyNode(g, axis_node, new_device_name, i);
                g->AddEdge(new_axis_node, 0, new_gather, 2);
              }
            }
            return Status::OK();
          }));
    }
  }

  if (prev_partition_num > 1) {
    TF_RETURN_IF_ERROR(MakeConcatOp(g, elastic_concat_node,
                                    elastic_concat_inputs, elastic_axis_node,
                                    concat_type, prev_partition_num));
  }
  for (auto& concat_meta : concat_metas) {
    TF_RETURN_IF_ERROR(MakeConcatOp(
        g, concat_meta.second.concat_node, concat_meta.second.concat_inputs,
        concat_meta.second.axis_node, concat_type, cur_partition_num));
  }

  return Status::OK();
}

void PartitionedVariable::Prepare(int cur_partition_nums,
                                  int prev_partition_num,
                                  int delta_partition_num) {
  sorted_opt_ev_names =
      std::vector<std::string>(opt_ev_names.begin(), opt_ev_names.end());
  // Make sure the opt variable is sorted by part.
  std::sort(sorted_opt_ev_names.begin(), sorted_opt_ev_names.end(),
            [](const std::string& str1, const std::string& str2) {
              auto part_idx = str1.rfind("/");
              std::string post_str = str1.substr(part_idx);
              auto post_idx = post_str.rfind("_");
              if (post_idx == string::npos) {
                return true;
              }

              auto part_idx_1 = str2.rfind("/");
              std::string post_str_1 = str2.substr(part_idx_1);
              auto post_idx_1 = post_str_1.rfind("_");
              if (post_idx_1 == string::npos) {
                return false;
              }

              return std::stoi(post_str.substr(post_idx + 1)) <
                     std::stoi(post_str_1.substr(post_idx_1 + 1));
            });

  cur_partition_num = cur_partition_nums;
  if ((var_type == DENSE_RESOUCE_VAR) || (var_type == DENSE_REF_VAR)) {
    // if (ev_partition_num + delta_partition_num <
    //     cur_partition_nums) {  // min-max-partitioner
    // cur_partition_num = ev_partition_num;
    // }
  }
}

Status PartitionedVariable::Scaling(int prev_partition_num) {
  Status s;
  VLOG(1) << DebugString();
  std::unordered_set<Node*> nodes_to_delete;
  if (ev_partition_num <= cur_partition_num) {
    s = ScalingUp(&nodes_to_delete, prev_partition_num);
  } else {
    s = ScalingDown(&nodes_to_delete);
  }
  for (auto* node : nodes_to_delete) {
    g->RemoveNode(node);
  }
  return s;
}

void PartitionedVariable::InitPartToDevice(
    const PartIdToNodeMap& partition_node_map, int& original_indices,
    std::vector<int>& part_to_device) {
  int part_id = -1;
  for (auto& part_node : partition_node_map) {
    part_to_device[part_node.first] = node_device_map[part_node.second];
    original_indices = part_node.first;
    part_id = std::max(node_device_map[part_node.second], part_id);
  }
  for (auto& part : part_to_device) {
    if (part == -1) {
      int assign_id = ++part_id % cur_partition_num;
      part = assign_id;
    }
  }
}

Status PartitionedVariable::ScalingUp(
    std::unordered_set<Node*>* nodes_to_delete, int prev_partition_num) {
  auto& partition_node_map = node_map[variable_prefix];
  std::vector<int> part_to_device(cur_partition_num, -1);
  int original_indices = 0;
  InitPartToDevice(partition_node_map, original_indices, part_to_device);

  switch (var_type) {
    case VarType::EMBEDDING_VAR:
      TF_RETURN_IF_ERROR(ScalingUpEmbeddingVar(original_indices, part_to_device,
                                               nodes_to_delete,
                                               prev_partition_num));
      break;
    case VarType::RESOURCE_VAR:
    case VarType::DENSE_RESOUCE_VAR:
      TF_RETURN_IF_ERROR(ScalingUpResVar(original_indices, part_to_device,
                                         nodes_to_delete, prev_partition_num));
      break;
    case VarType::REF_VAR:
    case VarType::DENSE_REF_VAR:
      TF_RETURN_IF_ERROR(ScalingUpVar(original_indices, part_to_device,
                                      nodes_to_delete, prev_partition_num));
      break;
  }

  return Status::OK();
}

Status PartitionedVariable::ScalingDown(
    std::unordered_set<Node*>* nodes_to_delete) {
  auto& partition_node_map = node_map[variable_prefix];
  std::vector<bool> part_to_scale_down(ev_partition_num, false);
  std::vector<int> part_to_move;

  for (auto& part_node : partition_node_map) {
    int device_id = node_device_map[part_node.second];
    if (device_id >= cur_partition_num) {
      part_to_scale_down[part_node.first] = true;
      part_to_move.emplace_back(part_node.first);
    }
  }
  
  if (part_to_move.size() == 0) return Status::OK();
  if (part_to_move.size() == cur_partition_num) {
    std::sort(part_to_move.begin(), part_to_move.end(),
              [](int a, int b) { return a < b; });
    MovePartitionedVariable(g, ev_partition_num, node_map, variable_prefix,
                            sorted_opt_ev_names, part_to_move, meta_node);
  } else {
    switch (var_type) {
      case VarType::EMBEDDING_VAR:
        TF_RETURN_IF_ERROR(ScalingDownEV(part_to_scale_down, nodes_to_delete));
        break;
      case VarType::RESOURCE_VAR:
      case VarType::DENSE_RESOUCE_VAR:
        TF_RETURN_IF_ERROR(
            ScalingDownResVar(part_to_scale_down, nodes_to_delete));
        break;
      case VarType::REF_VAR:
      case VarType::DENSE_REF_VAR:
        TF_RETURN_IF_ERROR(ScalingDownVar(part_to_scale_down, nodes_to_delete));
        break;
    }
  }
  return Status::OK();
}

Status PartitionedVariable::ScalingUpSparseVariableBackWardGraph(
    int original_indices,
    std::unordered_map<std::string, DynamicPartitionSubGraph>& prefix_name_map,
    std::vector<int>& part_to_device) {
  bool is_shared = prefix_name_map.size() > 1;
  Node* concat_node = nullptr;
  Node* axis_node = nullptr;
  DataType concat_type;
  std::vector<Node*> concat_inputs;
  for (auto& it : prefix_name_map) {
    Node* data_dp_node = it.second.data_dp_node;
    Node* indices_dp_node = it.second.indices_dp_node;
    for (int i = 0; i < cur_partition_num; ++i) {
      Node* cur_ev_node = node_map[variable_prefix][i];
      Node* cur_noop_node = meta_node->m_no_op_vec[i];
      if (i < ev_partition_num) {
        TF_RETURN_IF_ERROR(UpdateOldBackWardGraph(
            var_type, data_dp_node, indices_dp_node,
            sorted_opt_ev_names, g, i, cur_partition_num, &concat_node,
            &axis_node, concat_type, is_shared, it.first, concat_inputs,
            cur_ev_node));
      } else {
        Node* ev_node = node_map[variable_prefix][original_indices];
        string new_device_name = cur_ev_node->assigned_device_name();
        for (auto* node : ev_node->out_nodes()) {
          if (IsApplyNode(var_type, node)) {
            TF_RETURN_IF_ERROR(MakeSparseApplyOp(
                g, node, data_dp_node, indices_dp_node, cur_noop_node,
                variable_prefix, sorted_opt_ev_names, new_device_name, node_map,
                i, cur_partition_num, is_shared, it.first, concat_inputs));
          }
        }
      }
    }
  }
  MakeConcatOp(g, concat_node, concat_inputs, axis_node, concat_type,
               prefix_name_map.size());
  return Status::OK();
}

Status PartitionedVariable::ScalingUpDenseBackWardGraph(
    int original_indices,
    std::unordered_map<std::string, DynamicPartitionSubGraph>& prefix_name_map,
    std::vector<int>& part_to_device) {
  Node* var_node = node_map[variable_prefix][original_indices];
  Node* grad_concat_node = nullptr;
  Node* grad_axis_node = nullptr;
  std::vector<Node*> concat_nodes;
  std::vector<Node*> concat_outputs;
  for (int i = 0; i < cur_partition_num; ++i) {
    Node* cur_var_node = node_map[variable_prefix][i];
    Node* cur_noop_node = meta_node->m_no_op_vec[i];
    if (i < ev_partition_num) {
      TF_RETURN_IF_ERROR(UpdateOldDenseBackWardGraph(
          var_type, g, cur_var_node, &grad_concat_node, &grad_axis_node,
          &concat_nodes, &concat_outputs, part_var_full_shape, i,
          cur_partition_num, ev_partition_num, sorted_opt_ev_names,
          cur_noop_node));
    } else {
      string new_device_name = cur_var_node->assigned_device_name();
      for (auto* node : var_node->out_nodes()) {
        if (IsApplyNode(var_type, node)) {
          TF_RETURN_IF_ERROR(MakeApplyOp(
              g, node, cur_noop_node, &concat_nodes, &concat_outputs,
              variable_prefix, sorted_opt_ev_names, new_device_name, node_map,
              i, cur_partition_num, part_var_full_shape));
        }
      }
    }
  }

  TF_RETURN_IF_ERROR(MakeConcatOffsetOp(g, grad_concat_node, grad_axis_node,
                                        concat_nodes, concat_outputs,
                                        cur_partition_num));
  return Status::OK();
}

Status PartitionedVariable::ScalingDownSparseVariableBackWardGraph(
    std::unordered_set<Node*>* nodes_to_delete,
    std::unordered_map<std::string, DynamicPartitionSubGraph>& prefix_name_map,
    std::vector<bool>& part_to_scale_down) {
  bool is_shared = prefix_name_map.size() > 1;
  Node* concat_node = nullptr;
  Node* axis_node = nullptr;
  DataType concat_type;
  std::vector<Node*> concat_inputs;
  for (auto& it : prefix_name_map) {
    Node* data_dp_node = it.second.data_dp_node;
    Node* indices_dp_node = it.second.indices_dp_node;
    int offset = 0;
    for (int i = 0; i < part_to_scale_down.size(); ++i) {
      Node* cur_ev_node = node_map[variable_prefix][i];
      if (!part_to_scale_down[i]) {
        TF_RETURN_IF_ERROR(UpdateOldBackWardGraph(
            var_type, data_dp_node, indices_dp_node,
            sorted_opt_ev_names, g, i-offset, cur_partition_num, &concat_node,
            &axis_node, concat_type, is_shared, it.first, concat_inputs,
            cur_ev_node));
      } else {
        offset++;
        TF_RETURN_IF_ERROR(DeleteSparseBackWardGraph(
            var_type, cur_ev_node, sorted_opt_ev_names, nodes_to_delete));
      }
    }
  }
  MakeConcatOp(g, concat_node, concat_inputs, axis_node, concat_type,
               prefix_name_map.size());
  return Status::OK();
}

Status PartitionedVariable::ScalingDownDenseBackWardGraph(
    std::unordered_set<Node*>* nodes_to_delete,
    std::vector<bool>& part_to_scale_down) {
  int idx = 0;
  Node* grad_concat_node = nullptr;
  Node* grad_axis_node = nullptr;
  std::vector<Node*> concat_inputs;
  std::vector<Node*> concat_outputs;
  for (int i = 0; i < part_to_scale_down.size(); ++i) {
    Node* cur_var_node = node_map[variable_prefix][i];
    if (!part_to_scale_down[i]) {
      TF_RETURN_IF_ERROR(UpdateOldDenseBackWardGraph(
          var_type, g, cur_var_node, &grad_concat_node, &grad_axis_node,
          &concat_inputs, &concat_outputs, part_var_full_shape, idx++,
          cur_partition_num, ev_partition_num, sorted_opt_ev_names));
    } else {
      TF_RETURN_IF_ERROR(DeleteDenseBackWardGraph(
          var_type, g, cur_var_node, cur_partition_num, nodes_to_delete));
    }
  }
  TF_RETURN_IF_ERROR(MakeConcatOffsetOp(g, grad_concat_node, grad_axis_node,
                                        concat_inputs, concat_outputs,
                                        cur_partition_num));
  return Status::OK();
}

Status PartitionedVariable::RewriteDynamicPartitionGraph(
    PartIdToNodeMap& ev_node_vec,
    std::unordered_map<std::string, DynamicPartitionSubGraph>& prefix_name_map,
    std::unordered_set<Node*>* nodes_to_delete,
    const std::vector<bool>* part_to_scale_down) {
  TF_RETURN_IF_ERROR(InitDynamicPartitionGraphMeta(var_type, cur_partition_num,
                                                   ev_node_vec, prefix_name_map,
                                                   part_to_scale_down));
  for (auto& it : prefix_name_map) {
    auto& dynamic_partition_sub_graph = it.second;
    if ((dynamic_partition_sub_graph.dynamic_stitch_node == nullptr) ||
        (dynamic_partition_sub_graph.dynamic_partition_node == nullptr)) {
      continue;
    }
    TF_RETURN_IF_ERROR(MakeDynamicPartitionOp(var_type, cur_partition_num,
                                              dynamic_partition_sub_graph, g,
                                              nodes_to_delete, part_to_scale_down));

    TF_RETURN_IF_ERROR(MakeDynamicStitchOp(
        dynamic_partition_sub_graph, cur_partition_num, g, nodes_to_delete, part_to_scale_down));
  }

  return Status::OK();
}

Status PartitionedVariable::ScalingUpEmbeddingVar(
    int original_indices, std::vector<int>& part_to_device,
    std::unordered_set<Node*>* nodes_to_delete, int prev_partition_num) {
  TF_RETURN_IF_ERROR(ScalingUpEVForWardGraph(original_indices, part_to_device));

  std::vector<std::pair<Node*, Node*>> primary_ev_filters(cur_partition_num,
                                                          {nullptr, nullptr});

  TF_RETURN_IF_ERROR(ScalingUpEVRedistributionGraph(
      original_indices, node_map[variable_prefix], nodes_to_delete,
      part_to_device, primary_ev_filters));
  for (auto& opt_ev_name : sorted_opt_ev_names) {
    TF_RETURN_IF_ERROR(ScalingUpEVRedistributionGraph(
        original_indices, node_map[opt_ev_name], nodes_to_delete,
        part_to_device, primary_ev_filters));
  }

  std::unordered_map<std::string, DynamicPartitionSubGraph> prefix_name_map;

  TF_RETURN_IF_ERROR(RewriteDynamicPartitionGraph(
      node_map[variable_prefix], prefix_name_map, nodes_to_delete));

  TF_RETURN_IF_ERROR(ScalingUpSparseVariableBackWardGraph(
      original_indices, prefix_name_map, part_to_device));
  return Status::OK();
}

Status PartitionedVariable::ScalingUpResVar(
    int original_indices, std::vector<int>& part_to_device,
    std::unordered_set<Node*>* nodes_to_delete, int prev_partition_num) {
  TF_RETURN_IF_ERROR(ScalingUpResVarForWardGraph(
      original_indices, part_to_device, nodes_to_delete));

  TF_RETURN_IF_ERROR(ScalingUpResVarRedistributionGraph(
      original_indices, node_map[variable_prefix], part_to_device));
  for (auto& opt_ev_name : sorted_opt_ev_names) {
    TF_RETURN_IF_ERROR(ScalingUpResVarRedistributionGraph(
        original_indices, node_map[opt_ev_name], part_to_device));
  }

  std::unordered_map<std::string, DynamicPartitionSubGraph> prefix_name_map;

  TF_RETURN_IF_ERROR(RewriteDynamicPartitionGraph(
      node_map[variable_prefix], prefix_name_map, nodes_to_delete));

  if (var_type == VarType::RESOURCE_VAR) {
    TF_RETURN_IF_ERROR(ScalingUpSparseVariableBackWardGraph(
        original_indices, prefix_name_map, part_to_device));
  } else {
    TF_RETURN_IF_ERROR(ScalingUpDenseBackWardGraph(
        original_indices, prefix_name_map, part_to_device));
  }
  return Status::OK();
}

Status PartitionedVariable::ScalingUpVar(
    int original_indices, std::vector<int>& part_to_device,
    std::unordered_set<Node*>* nodes_to_delete, int prev_partition_num) {
  TF_RETURN_IF_ERROR(ScalingUpVarForWardGraph(
      original_indices, variable_prefix, part_to_device, prev_partition_num));
  for (auto& opt_ev_name : sorted_opt_ev_names) {
    TF_RETURN_IF_ERROR(ScalingUpVarForWardGraph(
        original_indices, opt_ev_name, part_to_device, prev_partition_num));
  }

  TF_RETURN_IF_ERROR(ScalingUpVarRedistributionGraph(
      original_indices, node_map[variable_prefix], part_to_device));
  for (auto& opt_ev_name : sorted_opt_ev_names) {
    TF_RETURN_IF_ERROR(ScalingUpVarRedistributionGraph(
        original_indices, node_map[opt_ev_name], part_to_device));
  }

  std::unordered_map<std::string, DynamicPartitionSubGraph> prefix_name_map;

  TF_RETURN_IF_ERROR(RewriteDynamicPartitionGraph(
      node_map[variable_prefix], prefix_name_map, nodes_to_delete));

  if (var_type == VarType::REF_VAR) {
    TF_RETURN_IF_ERROR(ScalingUpSparseVariableBackWardGraph(
        original_indices, prefix_name_map, part_to_device));
  } else {
    TF_RETURN_IF_ERROR(ScalingUpDenseBackWardGraph(
        original_indices, prefix_name_map, part_to_device));
  }
  return Status::OK();
}

Status PartitionedVariable::ScalingDownEV(
    std::vector<bool>& part_to_scale_down,
    std::unordered_set<Node*>* nodes_to_delete) {
  TF_RETURN_IF_ERROR(
      ScalingDownEVForWardGraph(part_to_scale_down, nodes_to_delete));

  std::vector<std::pair<Node*, Node*>> primary_ev_filters(ev_partition_num,
                                                          {nullptr, nullptr});

  TF_RETURN_IF_ERROR(ScalingDownEVRedistributionGraph(
      node_map[variable_prefix], primary_ev_filters, nodes_to_delete,
      part_to_scale_down));
  for (auto& opt_ev_name : sorted_opt_ev_names) {
    TF_RETURN_IF_ERROR(ScalingDownEVRedistributionGraph(
        node_map[opt_ev_name], primary_ev_filters, nodes_to_delete,
        part_to_scale_down));
  }

  std::unordered_map<std::string, DynamicPartitionSubGraph> prefix_name_map;

  TF_RETURN_IF_ERROR(
      RewriteDynamicPartitionGraph(node_map[variable_prefix], prefix_name_map,
                                   nodes_to_delete, &part_to_scale_down));

  TF_RETURN_IF_ERROR(ScalingDownSparseVariableBackWardGraph(
      nodes_to_delete, prefix_name_map, part_to_scale_down));
  return Status::OK();
}

Status PartitionedVariable::ScalingDownResVar(
    std::vector<bool>& part_to_scale_down,
    std::unordered_set<Node*>* nodes_to_delete) {
  auto& var_vec = node_map[variable_prefix];
  TF_RETURN_IF_ERROR(ScalingDownResVarForWardGraph(var_vec, part_to_scale_down,
                                                   nodes_to_delete));
  for (auto& opt_ev_name : sorted_opt_ev_names) {
    auto& tmp_var_node = node_map[opt_ev_name];
    TF_RETURN_IF_ERROR(ScalingDownResVarForWardGraph(
        tmp_var_node, part_to_scale_down, nodes_to_delete));
  }

  TF_RETURN_IF_ERROR(ScalingDownResVarRedistributionGraph(
      node_map[variable_prefix], nodes_to_delete, part_to_scale_down));
  for (auto& opt_ev_name : sorted_opt_ev_names) {
    TF_RETURN_IF_ERROR(ScalingDownResVarRedistributionGraph(
        node_map[opt_ev_name], nodes_to_delete, part_to_scale_down));
  }

  std::unordered_map<std::string, DynamicPartitionSubGraph> prefix_name_map;

  TF_RETURN_IF_ERROR(
      RewriteDynamicPartitionGraph(node_map[variable_prefix], prefix_name_map,
                                   nodes_to_delete, &part_to_scale_down));

  if (var_type == VarType::RESOURCE_VAR) {
    TF_RETURN_IF_ERROR(ScalingDownSparseVariableBackWardGraph(
        nodes_to_delete, prefix_name_map, part_to_scale_down));
  } else {
    TF_RETURN_IF_ERROR(
        ScalingDownDenseBackWardGraph(nodes_to_delete, part_to_scale_down));
  }
  return Status::OK();
}

Status PartitionedVariable::ScalingDownVar(
    std::vector<bool>& part_to_scale_down,
    std::unordered_set<Node*>* nodes_to_delete) {
  auto& var_vec = node_map[variable_prefix];
  TF_RETURN_IF_ERROR(
      ScalingDownVarForWardGraph(var_vec, part_to_scale_down, nodes_to_delete));
  for (auto& opt_ev_name : sorted_opt_ev_names) {
    auto& tmp_var_node = node_map[opt_ev_name];
    TF_RETURN_IF_ERROR(ScalingDownVarForWardGraph(
        tmp_var_node, part_to_scale_down, nodes_to_delete));
  }

  TF_RETURN_IF_ERROR(ScalingDownVarRedistributionGraph(var_vec, nodes_to_delete,
                                                       part_to_scale_down));
  for (auto& opt_ev_name : sorted_opt_ev_names) {
    TF_RETURN_IF_ERROR(ScalingDownVarRedistributionGraph(
        node_map[opt_ev_name], nodes_to_delete, part_to_scale_down));
  }

  std::unordered_map<std::string, DynamicPartitionSubGraph> prefix_name_map;

  TF_RETURN_IF_ERROR(
      RewriteDynamicPartitionGraph(node_map[variable_prefix], prefix_name_map,
                                   nodes_to_delete, &part_to_scale_down));

  if (var_type == VarType::REF_VAR) {
    TF_RETURN_IF_ERROR(ScalingDownSparseVariableBackWardGraph(
        nodes_to_delete, prefix_name_map, part_to_scale_down));
  } else {
    TF_RETURN_IF_ERROR(
        ScalingDownDenseBackWardGraph(nodes_to_delete, part_to_scale_down));
  }
  return Status::OK();
}

Status PartitionedVariable::ScalingUpEVRedistributionGraph(
    int original_indices, PartIdToNodeMap& ev_node_vec,
    std::unordered_set<Node*>* nodes_to_delete,
    std::vector<int>& part_to_device,
    std::vector<std::pair<Node*, Node*>>& primary_ev_filters) {
  // This function is responible for create new ev 's redistribution subgraph.
  Node* import_op_main = meta_node->m_import_op_main;
  DataType key_type, value_type;
  std::vector<Node*> filtered_node_vec;
  for (int i = 0; i < cur_partition_num; ++i) {
    TF_RETURN_IF_ERROR(MakeEVFilterOp(
        g, ev_node_vec, filtered_node_vec, primary_ev_filters, nodes_to_delete,
        key_type, value_type, ev_partition_num, i, cur_partition_num));
  }

  for (int i = 0; i < part_to_device.size(); ++i) {
    Node* ev_node = ev_node_vec[i];
    TF_RETURN_IF_ERROR(
        MakeEVMulImportOp(g, import_op_main, ev_node, filtered_node_vec,
                          primary_ev_filters, nodes_to_delete, key_type,
                          value_type, ev_partition_num, i, cur_partition_num));
  }

  return Status::OK();
}

Status PartitionedVariable::ScalingUpResVarRedistributionGraph(
    int original_indices, PartIdToNodeMap& var_node_vec,
    std::vector<int>& part_to_device) {
  Status s;
  Node* import_op_main = meta_node->m_import_op_main;
  Node* ori_var = var_node_vec[original_indices];
  Node* rhs_value_node = nullptr;
  Node* partition_num_node = nullptr;
  int partition_num;
  DataType key_type;
  for (auto* oo_node : ori_var->out_nodes()) {
    if (oo_node->type_string() == ::des::kReAssignRes) {
      TF_RETURN_IF_ERROR(
          GetNodeAttr(oo_node->attrs(), "partition_nums", &partition_num));
      TF_RETURN_IF_ERROR(oo_node->input_node(1, &rhs_value_node));
      TF_RETURN_IF_ERROR(oo_node->input_node(2, &partition_num_node));
      TF_RETURN_IF_ERROR(GetNodeAttr(oo_node->attrs(), "T", &key_type));
    }
  }

  for (int i = ev_partition_num; i < cur_partition_num; ++i) {
    auto* var_node = var_node_vec[i];
    NodeDef reassign_node_def;
    TF_RETURN_IF_ERROR(
        NodeDefBuilder(var_node->name() + "/ReAssignResource",
                       ::des::kReAssignRes)
            .Input(var_node->name(), 0, DT_RESOURCE)
            .Input(rhs_value_node->name(), 0, rhs_value_node->output_type(0))
            .Input(partition_num_node->name(), 0,
                   partition_num_node->output_type(0))
            .Attr("partition_id", i)
            .Attr("device_id", node_device_map[var_node])
            .Attr("partition_nums", partition_num)
            .Attr("T", key_type)
            .Device(var_node->assigned_device_name())
            .Finalize(&reassign_node_def));
    Node* reassign_node = g->AddNode(reassign_node_def, &s);
    TF_RETURN_IF_ERROR(s);
    reassign_node->set_assigned_device_name(var_node->assigned_device_name());
    g->AddControlEdge(reassign_node, import_op_main);
  }
  return s;
}

Status PartitionedVariable::ScalingUpVarRedistributionGraph(
    int original_indices, PartIdToNodeMap& var_node_vec,
    std::vector<int>& part_to_device) {
  Status s;
  Node* ori_var = var_node_vec[original_indices];
  Node* import_op_main = meta_node->m_import_op_main;
  bool use_locking;
  int partition_num;
  DataType key_type;
  Node* rhs_value_node = nullptr;
  Node* partition_num_node = nullptr;
  for (auto* oo_node : ori_var->out_nodes()) {
    if (oo_node->type_string() == ::des::kReAssign) {
      TF_RETURN_IF_ERROR(
          GetNodeAttr(oo_node->attrs(), "use_locking", &use_locking));
      TF_RETURN_IF_ERROR(
          GetNodeAttr(oo_node->attrs(), "partition_nums", &partition_num));
      TF_RETURN_IF_ERROR(GetNodeAttr(oo_node->attrs(), "T", &key_type));
      TF_RETURN_IF_ERROR(oo_node->input_node(1, &rhs_value_node));
      if (var_type == VarType::DENSE_REF_VAR) {
        TF_RETURN_IF_ERROR(MakeConstTensor(g, ori_var, cur_partition_num,
                                           &partition_num_node,
                                           ori_var->assigned_device_name()));
      } else {
        TF_RETURN_IF_ERROR(oo_node->input_node(2, &partition_num_node));
      }
      break;
    }
  }

  for (int i = 0; i < cur_partition_num; ++i) {
    auto* var_node = var_node_vec[i];
    if (i < ev_partition_num) {
      for (auto* oo_node : var_node->out_nodes()) {
        if (oo_node->type_string() == ::des::kReAssign) {
          TF_RETURN_IF_ERROR(g->UpdateEdge(partition_num_node, 0, oo_node, 2));
        }
      }
    } else {
      NodeDef reassign_node_def;
      TF_RETURN_IF_ERROR(
          NodeDefBuilder(var_node->name() + "/ReAssign", ::des::kReAssign)
              .Input(var_node->name(), 0, MakeRefType(key_type))
              .Input(rhs_value_node->name(), 0, rhs_value_node->output_type(0))
              .Input(partition_num_node->name(), 0,
                     partition_num_node->output_type(0))
              .Attr("use_locking", use_locking)
              .Attr("partition_id", i)
              .Attr("device_id", node_device_map[var_node])
              .Attr("partition_nums", partition_num)
              .Attr("T", key_type)
              .Device(var_node->assigned_device_name())
              .Finalize(&reassign_node_def));
      Node* reassign_node = g->AddNode(reassign_node_def, &s);
      TF_RETURN_IF_ERROR(s);
      reassign_node->set_assigned_device_name(var_node->assigned_device_name());
      g->AddControlEdge(reassign_node, import_op_main);
    }
  }
  return s;
}

Status PartitionedVariable::ScalingDownEVRedistributionGraph(
    PartIdToNodeMap& ev_node_vec,
    std::vector<std::pair<Node*, Node*>>& primary_ev_filters,
    std::unordered_set<Node*>* nodes_to_delete,
    std::vector<bool>& part_to_scale_down) {
  Status s;
  DataType key_type, value_type;
  std::vector<Node*> filtered_node_vec;
  Node* import_op_main = meta_node->m_import_op_main;
  for (int i = 0; i < part_to_scale_down.size(); ++i) {
    Node* ev_node = ev_node_vec[i];
    for (auto* o_node : ev_node->out_nodes()) {
      if (o_node->type_string() == ::des::kEvExportOp) {
        TF_RETURN_IF_ERROR(GetNodeAttr(o_node->attrs(), "Tkeys", &key_type));
        TF_RETURN_IF_ERROR(GetNodeAttr(o_node->attrs(), "dtype", &value_type));
        o_node->ClearAttr("partition_nums");
        o_node->AddAttr("partition_nums", cur_partition_num);
        if (!part_to_scale_down[i]) {
          filtered_node_vec.emplace_back(o_node);
        } else {
          nodes_to_delete->insert(o_node);
        }
      }
    }
  }
  int actual_offset = 0;
  for (int i = 0; i < part_to_scale_down.size(); ++i) {
    Node* ev_node = ev_node_vec[i];
    if (!part_to_scale_down[i]) {
      TF_RETURN_IF_ERROR(MakeEVMulImportOp(
          g, import_op_main, ev_node, filtered_node_vec, primary_ev_filters,
          nodes_to_delete, key_type, value_type, ev_partition_num, i - actual_offset,
          cur_partition_num));
    } else {
      actual_offset++;
      for (auto* o_node : ev_node->out_nodes()) {
        if (o_node->type_string() == ::des::kEvImportOp) {
          nodes_to_delete->insert(o_node);
        }
      }
    }
  }

  return s;
}

Status PartitionedVariable::ScalingDownResVarRedistributionGraph(
    PartIdToNodeMap& ev_node_vec, std::unordered_set<Node*>* nodes_to_delete,
    std::vector<bool>& part_to_scale_down) {
  for (int i = 0; i < part_to_scale_down.size(); ++i) {
    if (part_to_scale_down[i]) {
      auto* ev_node = ev_node_vec[i];
      for (auto* o_node : ev_node->out_nodes()) {
        if (o_node->type_string() == ::des::kReAssignRes) {
          nodes_to_delete->emplace(o_node);
        }
      }
    }
  }
  return Status::OK();
}

Status PartitionedVariable::ScalingDownVarRedistributionGraph(
    PartIdToNodeMap& ev_node_vec, std::unordered_set<Node*>* nodes_to_delete,
    std::vector<bool>& part_to_scale_down) {
  for (int i = 0; i < part_to_scale_down.size(); ++i) {
    if (part_to_scale_down[i]) {
      auto* ev_node = ev_node_vec[i];
      for (auto* o_node : ev_node->out_nodes()) {
        if (o_node->type_string() == kIdentityOp) {
          for (auto* oo_node : o_node->out_nodes()) {
            if (oo_node->type_string() == ::des::kReAssign) {
              nodes_to_delete->emplace(oo_node);
            }
          }
        }
      }
    }
  }

  return Status::OK();
}

REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_PLACEMENT, 0,
                      ElasticTrainingPass);

}  // namespace tensorflow
