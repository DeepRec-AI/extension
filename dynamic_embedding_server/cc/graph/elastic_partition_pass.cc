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

#include "include/json/json.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/util/dump_graph.h"
#include "tensorflow/core/util/env_var.h"

constexpr char kEnableElasticEnv[] = "ENABLE_DES";
constexpr char kEvInitOp[] = "InitializeKvVariableV2Op";
constexpr char kEvImportOp[] = "ImportStorage";
constexpr char kEvExportOp[] = "FilterStorage";
constexpr char kElasticImportScope[] = "elastic_import";
constexpr char kElasticSubGraphImport[] = "elastic_subgraph_import";
constexpr char kElasticSubGraphInit[] = "elastic_subgraph_init";
constexpr char kDatasetInit[] = "make_initializer";
constexpr char kReAssign[] = "ReAssign";
constexpr char kReAssignRes[] = "ReAssignResource";

namespace tensorflow {

int ElasticTrainingPass::cur_partition_nums_ = 0;

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
  bool enable_elastic_training = false;
  TF_RETURN_IF_ERROR(
      ReadBoolFromEnvVar(kEnableElasticEnv, false, &enable_elastic_training));

  if (!enable_elastic_training) {
    LOG(INFO) << "Elastic training not enable.";
    return Status::OK();
  }

  int partition_nums;
  TF_RETURN_IF_ERROR(UpdatePartitionNums(partition_nums));
  if (cur_partition_nums_ == 0) {
    cur_partition_nums_ = partition_nums;
    return Status::OK();
  } else if (cur_partition_nums_ == partition_nums) {
    LOG(INFO) << "No need to do elastic partition pass.";
    return Status::OK();
  } else {
    scaling_up_ = partition_nums > cur_partition_nums_;
    cur_partition_nums_ = partition_nums;
  }

  Graph* graph = options.graph->get();
  if (graph == nullptr) {
    return errors::Internal("a graph should be available.");
  }
  std::unique_ptr<Graph> new_graph(new Graph(OpRegistry::Global()));
  CopyGraph(*graph, new_graph.get());

  TF_RETURN_IF_ERROR(RewriteSubGraph(new_graph.get()));

  DumpGraphToFile("ElasticTraining", *new_graph.get(), options.flib_def);
  options.graph->swap(new_graph);
  return Status::OK();
}

Status ElasticTrainingPass::RewriteSubGraph(Graph* g, bool is_test) {
  std::unordered_map<std::string, PartitionVarMeta> primary_node_metas_map;
  std::unordered_map<std::string, std::vector<std::string>>
      primary_node_to_opt_map;
  std::unordered_map<std::string, std::vector<Node*>> node_to_origin_map;
  std::unordered_map<std::string, Node*> unpartitioned_node_map;
  ElasticHookMetaNode meta_node(cur_partition_nums_);
  TF_RETURN_IF_ERROR(InitHookMetaNode(g, meta_node));
  TF_RETURN_IF_ERROR(InitVarMeta(g, primary_node_metas_map,
                                 primary_node_to_opt_map, node_to_origin_map,
                                 unpartitioned_node_map));
  TF_RETURN_IF_ERROR(RewriteTrainingSubGraph(
      g, primary_node_metas_map, primary_node_to_opt_map, node_to_origin_map,
      meta_node, is_test));
  TF_RETURN_IF_ERROR(RewriteSavingSubGraph(
      g, primary_node_metas_map, primary_node_to_opt_map, node_to_origin_map,
      unpartitioned_node_map, meta_node));
  return Status::OK();
}

Status ElasticTrainingPass::ScalingDownForWardGraph(
    const VarType& var_type, Graph* g, int part_var_full_shape,
    int ev_partition_num, const std::string& primary_variable_name,
    const std::vector<std::string>& opt_ev_names,
    std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
    std::unordered_set<Node*>& nodes_to_delete) {
  Status s;
  switch (var_type) {
    case VarType::EMBEDDING_VAR:
      s = ScalingDownEVForWardGraph(var_type, g, ev_partition_num,
                                    primary_variable_name, opt_ev_names,
                                    node_to_origin_map, nodes_to_delete);
      break;
    case VarType::RESOURCE_VAR:
    case VarType::DENSE_RESOUCE_VAR:
      s = ScalingDownResVarForWardGraph(var_type, g, part_var_full_shape,
                                        ev_partition_num, primary_variable_name,
                                        opt_ev_names, node_to_origin_map,
                                        nodes_to_delete);
      break;
    default:
      s = ScalingDownVarForWardGraph(var_type, g, ev_partition_num,
                                     primary_variable_name, opt_ev_names,
                                     node_to_origin_map, nodes_to_delete);
      break;
  }
  return s;
}

Status ElasticTrainingPass::ScalingUpForWardGraph(
    const VarType& var_type, Graph* g, int part_var_full_shape,
    int ev_partition_num, const std::string& primary_variable_name,
    const std::vector<std::string>& opt_ev_names,
    ElasticHookMetaNode& meta_node,
    std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
    std::unordered_set<Node*>& nodes_to_delete) {
  Status s;
  switch (var_type) {
    case VarType::EMBEDDING_VAR:
      s = ScalingUpEVForWardGraph(
          var_type, g, ev_partition_num, primary_variable_name, opt_ev_names,
          meta_node, node_to_origin_map, nodes_to_delete);
      break;
    case VarType::RESOURCE_VAR:
    case VarType::DENSE_RESOUCE_VAR:
      s = ScalingUpResVarForWardGraph(var_type, g, part_var_full_shape,
                                      ev_partition_num, primary_variable_name,
                                      opt_ev_names, meta_node,
                                      node_to_origin_map, nodes_to_delete);
      break;
    default:
      s = ScalingUpVarForWardGraph(
          var_type, g, ev_partition_num, primary_variable_name, opt_ev_names,
          meta_node, node_to_origin_map, nodes_to_delete);
      break;
  }
  return s;
}

Status ElasticTrainingPass::ScalingDownEVForWardGraph(
    const VarType& var_type, Graph* g, int ev_partition_num,
    const std::string& primary_ev_name,
    const std::vector<std::string>& opt_ev_names,
    std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
    std::unordered_set<Node*>& nodes_to_delete) {
  for (int i = cur_partition_nums_; i < ev_partition_num; ++i) {
    Node* var_node = node_to_origin_map[primary_ev_name][i];
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

        nodes_to_delete.insert(o_node);
        const Edge* init_value_edge = nullptr;
        TF_RETURN_IF_ERROR(o_node->input_edge(2, &init_value_edge));
        nodes_to_delete.insert(init_value_edge->src());
        const Edge* empty_key_edge = nullptr;
        TF_RETURN_IF_ERROR(o_node->input_edge(3, &empty_key_edge));
        nodes_to_delete.insert(empty_key_edge->src());
      } else if (o_node->type_string() == "KvResourceGather") {
        nodes_to_delete.insert(o_node);
        for (auto* o_edge : o_node->out_edges()) {
          if (o_edge->dst()->type_string() == kIdentityOp) {
            nodes_to_delete.insert(o_edge->dst());
          }
        }
      }
    }

    for (auto& opt_ev_name : opt_ev_names) {
      Node* ev_node = node_to_origin_map[opt_ev_name][i];
      // nodes_to_delete.insert(ev_node);
      for (auto* o_node : ev_node->out_nodes()) {
        if (o_node->type_string() == kEvInitOp) {
          nodes_to_delete.insert(o_node);
          const Edge* init_value_edge = nullptr;
          TF_RETURN_IF_ERROR(o_node->input_edge(2, &init_value_edge));
          nodes_to_delete.insert(init_value_edge->src());
          const Edge* empty_key_edge = nullptr;
          TF_RETURN_IF_ERROR(o_node->input_edge(3, &empty_key_edge));
          nodes_to_delete.insert(empty_key_edge->src());
        }
      }
    }
  }
  return Status::OK();
}

Status ElasticTrainingPass::ScalingUpEVForWardGraph(
    const VarType& var_type, Graph* g, int ev_partition_num,
    const std::string& primary_variable_name,
    const std::vector<std::string>& opt_ev_names,
    ElasticHookMetaNode& meta_node,
    std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
    std::unordered_set<Node*>& nodes_to_delete) {
  // EVHandler
  auto& var_vec = node_to_origin_map[primary_variable_name];
  for (int i = ev_partition_num; i < cur_partition_nums_; ++i) {
    Node* cur_init_op = meta_node.m_init_op_vec[i];
    Node* ori_ev_node = var_vec[0];
    string new_device_name = NewDevice(ori_ev_node, i);
    std::string op_name =
        primary_variable_name + "/" + kPart + std::to_string(i);

    Node* new_ev_node = CopyNode(g, ori_ev_node, new_device_name, i, op_name);
    new_ev_node->ClearAttr("shared_name");
    new_ev_node->AddAttr("shared_name", op_name);
    new_ev_node->AddAttr("_class", {"loc:@" + op_name});
    var_vec[i] = new_ev_node;

    bool is_init = false;
    Node* primary_init_node;
    // InitializeEVResource
    TF_RETURN_IF_ERROR(FindNodeAndExec(
        ori_ev_node, kEvInitOp,
        [this, &primary_init_node, &g, &new_ev_node, &is_init, new_device_name,
         i](Node* target_node) {
          if (!is_init) {
            const Node* tmp_check_ev_0;
            TF_RETURN_IF_ERROR(target_node->input_node(0, &tmp_check_ev_0));
            const Node* tmp_check_ev_1;
            TF_RETURN_IF_ERROR(target_node->input_node(1, &tmp_check_ev_1));
            if (tmp_check_ev_0->name() == tmp_check_ev_1->name()) {
              is_init = true;
              primary_init_node = CopyNode(g, target_node, new_device_name, i);
              g->AddEdge(new_ev_node, 0, primary_init_node, 0);
              g->AddEdge(new_ev_node, 0, primary_init_node, 1);
              // init_value
              const Edge* init_value_edge = nullptr;
              TF_RETURN_IF_ERROR(target_node->input_edge(2, &init_value_edge));
              auto* init_value_node =
                  CopyNode(g, init_value_edge->src(), new_device_name, i);
              g->AddEdge(init_value_node, init_value_edge->src_output(),
                         primary_init_node, 2);

              // empty_key
              const Edge* empty_key_edge = nullptr;
              TF_RETURN_IF_ERROR(target_node->input_edge(3, &empty_key_edge));
              auto* empty_key_node =
                  CopyNode(g, empty_key_edge->src(), new_device_name, i);
              g->AddEdge(empty_key_node, empty_key_edge->src_output(),
                         primary_init_node, 3);
            }
          }
          return Status::OK();
        }));

    g->AddControlEdge(primary_init_node, cur_init_op);

    TF_RETURN_IF_ERROR(FindNodeAndExec(
        ori_ev_node, "KvResourceGather",
        [this, &g, &new_ev_node, new_device_name, i](Node* target_node) {
          Node* gather_op = CopyNode(g, target_node, new_device_name, i);
          g->AddEdge(new_ev_node, 0, gather_op, 0);
          const Edge* gather_id_edge = nullptr;
          TF_RETURN_IF_ERROR(target_node->input_edge(1, &gather_id_edge));
          g->AddEdge(gather_id_edge->src(), gather_id_edge->src_output(),
                     gather_op, 1);
          const Edge* axis_edge = nullptr;
          TF_RETURN_IF_ERROR(target_node->input_edge(2, &axis_edge));
          Node* axis = CopyNode(g, axis_edge->src(), new_device_name, i);
          g->AddEdge(axis, 0, gather_op, 2);
          for (auto* o_edge : target_node->out_edges()) {
            if (o_edge->dst()->type_string() == kIdentityOp) {
              Node* identity_op =
                  CopyNode(g, o_edge->dst(), new_device_name, i);
              g->AddEdge(gather_op, 0, identity_op, 0);
            }
          }
          return Status::OK();
        }));

    // OptEV
    for (auto& opt_ev_name : opt_ev_names) {
      auto opt_var_node = node_to_origin_map[opt_ev_name][0];
      auto sep_idx = opt_ev_name.rfind("/");
      std::string op_name = opt_ev_name.substr(0, sep_idx) + "/" + kPart +
                            std::to_string(i) + opt_ev_name.substr(sep_idx);

      // EVHandler
      Node* new_opt_ev_node =
          CopyNode(g, opt_var_node, new_device_name, i, op_name);
      new_opt_ev_node->ClearAttr("shared_name");
      new_opt_ev_node->AddAttr("shared_name", op_name);
      node_to_origin_map[opt_ev_name][i] = new_opt_ev_node;

      is_init = false;
      TF_RETURN_IF_ERROR(FindNodeAndExec(
          opt_var_node, kEvInitOp,
          [this, &g, &primary_init_node, &new_ev_node, &cur_init_op, &is_init,
           &new_opt_ev_node, new_device_name, i](Node* target_node) {
            if (!is_init) {
              is_init = true;
              Node* init_node = CopyNode(g, target_node, new_device_name, i);
              g->AddEdge(new_opt_ev_node, 0, init_node, 0);
              g->AddEdge(new_ev_node, 0, init_node, 1);
              g->AddControlEdge(primary_init_node, init_node);
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

              g->AddEdge(empty_key_node, empty_key_edge->src_output(),
                         init_node, 3);
            }
            return Status::OK();
          }));
    }
  }
  return Status::OK();
}

Status ElasticTrainingPass::ScalingDownResVarForWardGraph(
    const VarType& var_type, Graph* g, int part_var_full_shape,
    int ev_partition_num, const std::string& primary_variable_name,
    const std::vector<std::string>& opt_ev_names,
    std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
    std::unordered_set<Node*>& nodes_to_delete) {
  for (int i = 0; i < ev_partition_num; ++i) {
    Node* ev_node = node_to_origin_map[primary_variable_name][i];
    if (i < cur_partition_nums_) {
      TensorShape new_shape;
      TF_RETURN_IF_ERROR(ChangeResShape(ev_node, new_shape, part_var_full_shape,
                                        cur_partition_nums_, false));

      for (auto* o_node : ev_node->out_nodes()) {
        if (o_node->type_string() == kIdentityOp) {
          for (auto* o_edge : o_node->out_edges()) {
            // Normal Variable
            if (o_edge->dst()->type_string() == "ConcatV2") {
              int N;
              TF_RETURN_IF_ERROR(GetNodeAttr(o_edge->dst()->attrs(), "N", &N));
              if (N != cur_partition_nums_) {
                const Edge* axis_edge = nullptr;
                TF_RETURN_IF_ERROR(o_edge->dst()->input_edge(N, &axis_edge));
                g->AddEdge(axis_edge->src(), 0, o_edge->dst(),
                           cur_partition_nums_);
                g->RemoveEdge(axis_edge);
                o_edge->dst()->ClearAttr("N");
                o_edge->dst()->AddAttr("N", cur_partition_nums_);
              }
              if (o_edge->dst_input() != i) {
                g->AddEdge(o_node, 0, o_edge->dst(), i);
                g->RemoveEdge(o_edge);
              }
            }
          }
        }
      }
      for (auto& opt_ev_name : opt_ev_names) {
        Node* opt_node = node_to_origin_map[opt_ev_name][i];
        TensorShape new_shape;
        TF_RETURN_IF_ERROR(ChangeResShape(opt_node, new_shape,
                                          part_var_full_shape,
                                          cur_partition_nums_, false));

        for (auto* o_node : opt_node->out_nodes()) {
          if (o_node->type_string() == kIdentityOp) {
            for (auto* o_edge : o_node->out_edges()) {
              if (o_edge->dst()->type_string() == "ConcatV2") {
                int N;
                TF_RETURN_IF_ERROR(
                    GetNodeAttr(o_edge->dst()->attrs(), "N", &N));
                if (N != cur_partition_nums_) {
                  const Edge* axis_edge = nullptr;
                  TF_RETURN_IF_ERROR(o_edge->dst()->input_edge(N, &axis_edge));
                  g->AddEdge(axis_edge->src(), 0, o_edge->dst(),
                             cur_partition_nums_);
                  g->RemoveEdge(axis_edge);
                  o_edge->dst()->ClearAttr("N");
                  o_edge->dst()->AddAttr("N", cur_partition_nums_);
                }
                if (o_edge->dst_input() != i) {
                  g->AddEdge(o_node, 0, o_edge->dst(), i);
                  g->RemoveEdge(o_edge);
                }
              }
            }
          }
        }
      }
    } else {
      for (auto* o_node : ev_node->out_nodes()) {
        if (o_node->type_string() == "ResourceGather") {
          nodes_to_delete.insert(o_node);
          for (auto* oo_node : o_node->out_nodes()) {
            if (oo_node->type_string() == kIdentityOp) {
              nodes_to_delete.insert(oo_node);
            }
          }
        } else if (o_node->type_string() == "AssignVariableOp") {
          nodes_to_delete.insert(o_node);
        }
      }
      for (auto& opt_ev_name : opt_ev_names) {
        Node* opt_node = node_to_origin_map[opt_ev_name][i];
        for (auto* o_node : opt_node->out_nodes()) {
          if (o_node->type_string() == "AssignVariableOp") {
            nodes_to_delete.insert(o_node);
          }
        }
      }
    }
  }
  return Status::OK();
}

Status ElasticTrainingPass::ScalingUpResVarForWardGraph(
    const VarType& var_type, Graph* g, int part_var_full_shape,
    int ev_partition_num, const std::string& primary_variable_name,
    const std::vector<std::string>& opt_ev_names,
    ElasticHookMetaNode& meta_node,
    std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
    std::unordered_set<Node*>& nodes_to_delete) {
  auto& var_vec = node_to_origin_map[primary_variable_name];
  for (int i = 0; i < cur_partition_nums_; ++i) {
    if (i < ev_partition_num) {
      Node* var_node = var_vec[i];
      TensorShape var_shape;
      TF_RETURN_IF_ERROR(ChangeResShape(var_node, var_shape,
                                        part_var_full_shape,
                                        cur_partition_nums_, false));

      for (auto& opt_ev_name : opt_ev_names) {
        auto opt_var_node = node_to_origin_map[opt_ev_name][i];
        TF_RETURN_IF_ERROR(ChangeResShape(opt_var_node, var_shape,
                                          part_var_full_shape,
                                          cur_partition_nums_, false));
      }
    } else {
      Node* var_node = var_vec[0];
      Node* cur_init_op = meta_node.m_init_op_vec[i];
      string new_device_name = NewDevice(var_node, i);
      std::string op_name =
          primary_variable_name + "/" + kPart + std::to_string(i);
      Node* new_var_node = CopyNode(g, var_node, new_device_name, i, op_name);
      TensorShape new_shape;
      TF_RETURN_IF_ERROR(
          ChangeResShape(new_var_node, new_shape, part_var_full_shape,
                         cur_partition_nums_, i == cur_partition_nums_ - 1));

      var_vec[i] = new_var_node;

      TF_RETURN_IF_ERROR(FindNodeAndExec(
          var_node, "AssignVariableOp",
          [this, &g, &new_var_node, &cur_init_op, new_shape, new_device_name,
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
          [this, &g, &new_var_node, new_device_name, i](Node* target_node) {
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
          [this, &g, &new_var_node, new_device_name, i](Node* target_node) {
            Node* new_var_read = CopyNode(g, target_node, new_device_name, i);
            g->AddEdge(new_var_node, 0, new_var_read, 0);
            for (auto* oo_node : target_node->out_nodes()) {
              // Normal Variable
              if ((oo_node->type_string() == "ConcatV2") &&
                  (oo_node->name().find(kElasticImportScope) == string::npos)) {
                int N;
                TF_RETURN_IF_ERROR(GetNodeAttr(oo_node->attrs(), "N", &N));
                if (N != cur_partition_nums_) {
                  const Edge* axis_edge;
                  TF_RETURN_IF_ERROR(oo_node->input_edge(N, &axis_edge));
                  oo_node->ClearAttr("N");
                  oo_node->AddAttr("N", cur_partition_nums_);
                  g->RemoveEdge(axis_edge);
                  g->AddEdge(axis_edge->src(), 0, oo_node, cur_partition_nums_);
                }
                g->AddEdge(new_var_read, 0, oo_node, i);
              }
            }
            return Status::OK();
          }));

      for (auto& opt_ev_name : opt_ev_names) {
        auto var_node = node_to_origin_map[opt_ev_name][0];
        auto sep_idx = opt_ev_name.rfind("/");
        std::string op_name = opt_ev_name.substr(0, sep_idx) + "/" + kPart +
                              std::to_string(i) + opt_ev_name.substr(sep_idx);

        Node* new_opt_var_node =
            CopyNode(g, var_node, new_device_name, i, op_name);
        TensorShape new_shape;
        TF_RETURN_IF_ERROR(
            ChangeResShape(new_opt_var_node, new_shape, part_var_full_shape,
                           cur_partition_nums_, i == cur_partition_nums_ - 1));

        node_to_origin_map[opt_ev_name][i] = new_opt_var_node;

        TF_RETURN_IF_ERROR(FindNodeAndExec(
            var_node, "AssignVariableOp",
            [this, &g, &new_opt_var_node, &cur_init_op, new_shape,
             new_device_name, i](Node* target_node) {
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
            [this, &g, &new_opt_var_node, new_device_name,
             i](Node* target_node) {
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
                  if (N != cur_partition_nums_) {
                    const Edge* axis_edge;
                    TF_RETURN_IF_ERROR(oo_node->input_edge(N, &axis_edge));
                    oo_node->ClearAttr("N");
                    oo_node->AddAttr("N", cur_partition_nums_);
                    g->RemoveEdge(axis_edge);
                    g->AddEdge(axis_edge->src(), 0, oo_node,
                               cur_partition_nums_);
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

Status ElasticTrainingPass::ScalingDownVarForWardGraph(
    const VarType& var_type, Graph* g, int ev_partition_num,
    const std::string& primary_variable_name,
    const std::vector<std::string>& opt_ev_names,
    std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
    std::unordered_set<Node*>& nodes_to_delete) {
  for (int i = cur_partition_nums_; i < ev_partition_num; ++i) {
    Node* ev_node = node_to_origin_map[primary_variable_name][i];
    for (auto* o_node : ev_node->out_nodes()) {
      if (o_node->type_string() == kIdentityOp) {
        nodes_to_delete.insert(o_node);
        for (auto* oo_node : o_node->out_nodes()) {
          if (oo_node->type_string() == "GatherV2") {
            nodes_to_delete.insert(oo_node);
          }
        }
      } else if (o_node->type_string() == "Assign") {
        nodes_to_delete.insert(o_node);
      }
    }
    for (auto& opt_ev_name : opt_ev_names) {
      Node* opt_node = node_to_origin_map[opt_ev_name][i];
      for (auto* o_node : opt_node->out_nodes()) {
        if (o_node->type_string() == kIdentityOp) {
          nodes_to_delete.insert(o_node);
        } else if (o_node->type_string() == "Assign") {
          nodes_to_delete.insert(o_node);
        }
      }
    }
  }
  return Status::OK();
}

Status ElasticTrainingPass::ScalingUpVarForWardGraph(
    const VarType& var_type, Graph* g, int ev_partition_num,
    const std::string& primary_variable_name,
    const std::vector<std::string>& opt_ev_names,
    ElasticHookMetaNode& meta_node,
    std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
    std::unordered_set<Node*>& nodes_to_delete) {
  auto& var_vec = node_to_origin_map[primary_variable_name];
  Node* var_node = var_vec[0];

  for (int i = ev_partition_num; i < cur_partition_nums_; ++i) {
    Node* cur_init_op = meta_node.m_init_op_vec[i];
    string new_device_name = NewDevice(var_node, i);
    std::string op_name =
        primary_variable_name + "/" + kPart + std::to_string(i);
    Node* new_var_node = CopyNode(g, var_node, new_device_name, i, op_name);
    var_vec[i] = new_var_node;
    TF_RETURN_IF_ERROR(FindNodeAndExec(
        var_node, "Assign",
        [this, &g, &new_var_node, &cur_init_op, new_device_name,
         i](Node* target_node) {
          if (target_node->name().find("save") == -1) {
            Node* new_var_init = CopyNode(g, target_node, new_device_name, i);
            g->AddEdge(new_var_node, 0, new_var_init, 0);

            const Edge* init_value_edge = nullptr;
            TF_RETURN_IF_ERROR(target_node->input_edge(1, &init_value_edge));
            Node* new_init_value =
                CopyNode(g, init_value_edge->src(), new_device_name, i);
            g->AddEdge(new_init_value, 0, new_var_init, 1);
            g->AddControlEdge(new_var_init, cur_init_op);
          }
          return Status::OK();
        }));

    TF_RETURN_IF_ERROR(FindNodeAndExec(
        var_node, kIdentityOp,
        [this, &g, &new_var_node, &var_type, new_device_name,
         i](Node* target_node) {
          Node* new_var_read = CopyNode(g, target_node, new_device_name, i);
          g->AddEdge(new_var_node, 0, new_var_read, 0);
          for (auto* oo_node : target_node->out_nodes()) {
            // Normal Variable
            if (oo_node->type_string() == "ConcatV2") {
              if (oo_node->name().find(kElasticImportScope) != string::npos) {
                continue;
              }
              // exactly once
              int N;
              TF_RETURN_IF_ERROR(GetNodeAttr(oo_node->attrs(), "N", &N));
              if (N != cur_partition_nums_) {
                const Edge* axis_edge;
                TF_RETURN_IF_ERROR(oo_node->input_edge(N, &axis_edge));
                oo_node->ClearAttr("N");
                oo_node->AddAttr("N", cur_partition_nums_);
                g->RemoveEdge(axis_edge);
                g->AddEdge(axis_edge->src(), 0, oo_node, cur_partition_nums_);
              }
              g->AddEdge(new_var_read, 0, oo_node, i);
            } else if (oo_node->type_string() == "GatherV2") {
              Node* new_gather = CopyNode(g, oo_node, new_device_name, i);
              g->AddEdge(new_var_read, 0, new_gather, 0);
              Node* axis_node = nullptr;
              TF_RETURN_IF_ERROR(oo_node->input_node(2, &axis_node));
              Node* new_axis_node = CopyNode(g, axis_node, new_device_name, i);
              g->AddEdge(new_axis_node, 0, new_gather, 2);
            }
          }
          return Status::OK();
        }));

    for (auto& opt_ev_name : opt_ev_names) {
      auto var_node = node_to_origin_map[opt_ev_name][0];
      auto sep_idx = opt_ev_name.rfind("/");
      std::string op_name = opt_ev_name.substr(0, sep_idx) + "/" + kPart +
                            std::to_string(i) + opt_ev_name.substr(sep_idx);

      Node* new_opt_var_node =
          CopyNode(g, var_node, new_device_name, i, op_name);

      node_to_origin_map[opt_ev_name][i] = new_opt_var_node;

      TF_RETURN_IF_ERROR(FindNodeAndExec(
          var_node, "Assign",
          [this, &g, &new_opt_var_node, &cur_init_op, new_device_name,
           i](Node* target_node) {
            if (target_node->name().find("save") == -1) {
              Node* new_var_init = CopyNode(g, target_node, new_device_name, i);
              g->AddEdge(new_opt_var_node, 0, new_var_init, 0);

              const Edge* init_value_edge = nullptr;
              TF_RETURN_IF_ERROR(target_node->input_edge(1, &init_value_edge));
              Node* new_const_init =
                  CopyNode(g, init_value_edge->src(), new_device_name, i);
              g->AddEdge(new_const_init, 0, new_var_init, 1);
              g->AddControlEdge(new_var_init, cur_init_op);
            }
            return Status::OK();
          }));

      TF_RETURN_IF_ERROR(FindNodeAndExec(
          var_node, kIdentityOp,
          [this, &g, &new_opt_var_node, new_device_name, i](Node* target_node) {
            Node* new_opt_var_read =
                CopyNode(g, target_node, new_device_name, i);
            g->AddEdge(new_opt_var_node, 0, new_opt_var_read, 0);
            for (auto* oo_node : target_node->out_nodes()) {
              // Normal Variable
              if (oo_node->type_string() == "ConcatV2") {
                if (oo_node->name().find(kElasticImportScope) != string::npos) {
                  continue;
                }
                int N;
                TF_RETURN_IF_ERROR(GetNodeAttr(oo_node->attrs(), "N", &N));
                if (N != cur_partition_nums_) {
                  const Edge* axis_edge;
                  TF_RETURN_IF_ERROR(oo_node->input_edge(N, &axis_edge));
                  oo_node->ClearAttr("N");
                  oo_node->AddAttr("N", cur_partition_nums_);
                  g->AddEdge(axis_edge->src(), 0, oo_node, cur_partition_nums_);
                }
                TF_RETURN_IF_ERROR(
                    g->UpdateEdge(new_opt_var_read, 0, oo_node, i));
              }
            }
            return Status::OK();
          }));
    }
  }
  return Status::OK();
}

Status ElasticTrainingPass::RewriteSavingSubGraph(
    Graph* g, std::unordered_map<std::string, PartitionVarMeta>& primary_node_metas_map,
    std::unordered_map<std::string, std::vector<std::string>>&
        primary_node_to_opt_map,
    std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
    std::unordered_map<std::string, Node*>& unpartitioned_node_map,
    ElasticHookMetaNode& meta_node) {
  std::vector<Node*> save_node_vec;
  std::vector<Node*> restore_node_vec;
  for (auto* node : g->nodes()) {
    if (node->type_string() == kSaveOp) {
      save_node_vec.emplace_back(node);
    } else if(node->type_string() == kRestoreOp) {
      restore_node_vec.emplace_back(node);
    }
  }

  if ((save_node_vec.size() == 0) || (restore_node_vec.size() == 0)) {
    LOG(INFO) << "There is no SaveV3 and RestoreV2 Op in Graph, Nothing to do.";
    return Status::OK();
  }

  std::unordered_set<Node*> nodes_to_delete;
  std::unordered_map<string, std::vector<int64>> partitioned_variable_shape;
  std::unordered_map<Node*, std::pair<string, string>> nodes_to_add;
  std::unordered_set<Node*> eval_nodes_to_add;

  TF_RETURN_IF_ERROR(GetPartVariableShape(primary_node_metas_map,
                                      save_node_vec[0],
                                      partitioned_variable_shape, cur_partition_nums_));

  if (scaling_up_) {
    for (int i = 0; i < cur_partition_nums_ ; ++i) {
      if (i < save_node_vec.size()) {
        TF_RETURN_IF_ERROR(RewritePrevSubGraph(g, i, cur_partition_nums_, scaling_up_, save_node_vec,
                                               node_to_origin_map, primary_node_metas_map,
                                               partitioned_variable_shape, nodes_to_add,
                                               nodes_to_delete, eval_nodes_to_add));
      } else {
        Status s;
        std::vector<Node*> restore_tensor_vec;
        std::vector<string> tensor_names_vec;
        std::vector<DataType> n_dtypes;
        bool has_ev = false;
        std::string assigned_device_name = "";
        Node* new_sharded_filename;
        
        TF_RETURN_IF_ERROR(AddNewSaverGraph(g, has_ev, &new_sharded_filename, tensor_names_vec, n_dtypes,
                                            primary_node_metas_map, node_to_origin_map, partitioned_variable_shape,
                                            restore_tensor_vec, assigned_device_name, 
                                            save_node_vec, i, cur_partition_nums_));
        TF_RETURN_IF_ERROR(AddNewRestoreGraph(g, has_ev, i, cur_partition_nums_, new_sharded_filename,
                                              tensor_names_vec, partitioned_variable_shape,
                                              restore_tensor_vec, restore_node_vec,
                                              assigned_device_name, n_dtypes));
      }
    }
  } else if (save_node_vec.size() > cur_partition_nums_) {
    for (int i = save_node_vec.size() - 1; i >=0 ; --i) {
      if (i < cur_partition_nums_) {
        TF_RETURN_IF_ERROR(RewritePrevSubGraph(g, i, cur_partition_nums_, scaling_up_, save_node_vec,
                                               node_to_origin_map, primary_node_metas_map,
                                               partitioned_variable_shape, nodes_to_add,
                                               nodes_to_delete, eval_nodes_to_add));
      } else {
        TF_RETURN_IF_ERROR(DeleteOldSaverGraph(save_node_vec, partitioned_variable_shape,
                                               unpartitioned_node_map, i, cur_partition_nums_,
                                               nodes_to_add, nodes_to_delete));
      }
    }

    for (auto& it: nodes_to_add) {
      TF_RETURN_IF_ERROR(MoveUnPartitionedVariable(g, it.first, meta_node));
    }

    for (auto& it: eval_nodes_to_add) {
      DeleteUnlessUnPartitionedVariable(g, it);
    }
  }

  for (auto* n : nodes_to_delete) {
    g->RemoveNode(n);
  }

  return Status::OK();
}

Status ElasticTrainingPass::RewriteTrainingSubGraph(
    Graph* g,
    std::unordered_map<std::string, PartitionVarMeta>& primary_node_metas_map,
    std::unordered_map<std::string, std::vector<std::string>>&
        primary_node_to_opt_map,
    std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
    ElasticHookMetaNode& meta_node, bool is_test) {
  std::vector<Node*> no_op_vec(cur_partition_nums_, nullptr);

  for (auto it : primary_node_to_opt_map) {
    std::unordered_set<Node*> nodes_to_delete;

    auto& primary_variable_name = it.first;
    auto& opt_ev_names = it.second;
    VarType var_type = primary_node_metas_map[primary_variable_name].m_var_type;
    int part_var_full_shape =
        primary_node_metas_map[primary_variable_name].m_full_shape;
    int ev_partition_num =
        primary_node_metas_map[primary_variable_name].m_partition_num;
    auto& var_vec = node_to_origin_map[primary_variable_name];

    // Make sure the opt variable is sorted by part.
    std::sort(opt_ev_names.begin(), opt_ev_names.end(),
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

                return std::stoi(post_str.substr(post_idx)) <
                       std::stoi(post_str_1.substr(post_idx_1));
              });

    LOG(INFO) << "processing: " << primary_variable_name << " var_type "
              << var_type << " var_vec size: " << var_vec.size();

    Node* elastic_node;
    Node* p_dynamic_stitch_node;

    // TODO(JUNQI) : per variable placement strategy
    if ((ev_partition_num == cur_partition_nums_) || (ev_partition_num == 1)) {
      LOG(INFO) << "Skip current variable.";
      continue;
    } else if (ev_partition_num < cur_partition_nums_) {
      TF_RETURN_IF_ERROR(ScalingUpForWardGraph(
          var_type, g, part_var_full_shape, ev_partition_num,
          primary_variable_name, opt_ev_names, meta_node, node_to_origin_map,
          nodes_to_delete));

      std::vector<Node*> primary_ev_filters(cur_partition_nums_, nullptr);
      TF_RETURN_IF_ERROR(ScalingUpRedistributionGraph(
          var_type, g, var_vec, meta_node.m_import_op_main, ev_partition_num,
          primary_ev_filters));
      for (auto& opt_ev_name : opt_ev_names) {
        TF_RETURN_IF_ERROR(ScalingUpRedistributionGraph(
            var_type, g, node_to_origin_map[opt_ev_name],
            meta_node.m_import_op_main, ev_partition_num, primary_ev_filters));
      }
      TF_RETURN_IF_ERROR(RewriteElasticPartitionGraph(
          var_type, g, node_to_origin_map[primary_variable_name], &elastic_node,
          &p_dynamic_stitch_node, nodes_to_delete));
      TF_RETURN_IF_ERROR(ScalingUpBackWardGraph(
          var_type, g, node_to_origin_map, primary_variable_name, opt_ev_names,
          elastic_node, p_dynamic_stitch_node, no_op_vec, part_var_full_shape,
          ev_partition_num));
    } else {  // scale down

      TF_RETURN_IF_ERROR(ScalingDownForWardGraph(
          var_type, g, part_var_full_shape, ev_partition_num,
          primary_variable_name, opt_ev_names, node_to_origin_map,
          nodes_to_delete));

      TF_RETURN_IF_ERROR(ScalingDownRedistributionGraph(
          var_type, g, var_vec, nodes_to_delete, ev_partition_num));
      for (auto& opt_ev_name : opt_ev_names) {
        TF_RETURN_IF_ERROR(ScalingDownRedistributionGraph(
            var_type, g, node_to_origin_map[opt_ev_name], nodes_to_delete,
            ev_partition_num));
      }

      TF_RETURN_IF_ERROR(RewriteElasticPartitionGraph(
          var_type, g, node_to_origin_map[primary_variable_name], &elastic_node,
          &p_dynamic_stitch_node, nodes_to_delete));

      TF_RETURN_IF_ERROR(ScalingDownBackWardGraph(
          g, var_type, node_to_origin_map, nodes_to_delete,
          primary_variable_name, opt_ev_names, elastic_node,
          p_dynamic_stitch_node, part_var_full_shape, ev_partition_num));
    }

    for (auto* node : nodes_to_delete) {
      g->RemoveNode(node);
    }
  }

  return Status::OK();
}

Status ElasticTrainingPass::ScalingUpEmbeddingVariableBackWardGraph(
    VarType var_type, Graph* g,
    std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
    const std::string& primary_ev_name,
    const std::vector<std::string>& opt_ev_names, Node* elastic_node,
    Node* p_dynamic_stitch_node, std::vector<Node*>& no_op_vec,
    int part_var_full_shape, int ev_partition_num) {
  Node* ev_node = node_to_origin_map[primary_ev_name][0];
  for (int i = 0; i < cur_partition_nums_; ++i) {
    Node* cur_ev_node = node_to_origin_map[primary_ev_name][i];
    Node* cur_noop_node = no_op_vec[i];
    if (i < ev_partition_num) {
      TF_RETURN_IF_ERROR(UpdateOldBackWardGraph(var_type, cur_ev_node,
                                                elastic_node, opt_ev_names, g,
                                                i, cur_partition_nums_));
    } else {
      string new_device_name = cur_ev_node->assigned_device_name();
      for (auto* node : ev_node->out_nodes()) {
        if (IsApplyNode(var_type, node)) {
          TF_RETURN_IF_ERROR(
              MakeNoOp(g, cur_noop_node, node, new_device_name, no_op_vec, i));
          TF_RETURN_IF_ERROR(
              MakeSparseApplyOp(g, node, elastic_node, cur_noop_node,
                                primary_ev_name, opt_ev_names, new_device_name,
                                node_to_origin_map, i, cur_partition_nums_));
        }
      }
    }
  }

  return Status::OK();
}

Status ElasticTrainingPass::ScalingUpDenseBackWardGraph(
    VarType var_type, Graph* g,
    std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
    const std::string& primary_ev_name,
    const std::vector<std::string>& opt_ev_names, Node* elastic_node,
    std::vector<Node*>& no_op_vec, int part_var_full_shape,
    int ev_partition_num) {
  Node* var_node = node_to_origin_map[primary_ev_name][0];
  for (int i = 0; i < cur_partition_nums_; ++i) {
    Node* cur_var_node = node_to_origin_map[primary_ev_name][i];
    if (i < ev_partition_num) {
      TF_RETURN_IF_ERROR(UpdateOldDenseBackWardGraph(var_type, g, cur_var_node,
                                                     part_var_full_shape, i,
                                                     cur_partition_nums_));
    } else {
      Node* cur_noop_node = no_op_vec[i];
      string new_device_name = cur_var_node->assigned_device_name();
      for (auto* node : var_node->out_nodes()) {
        if (IsApplyNode(var_type, node)) {
          TF_RETURN_IF_ERROR(
              MakeNoOp(g, cur_noop_node, node, new_device_name, no_op_vec, i));
          TF_RETURN_IF_ERROR(
              MakeApplyOp(g, node, cur_noop_node, primary_ev_name, opt_ev_names,
                          new_device_name, node_to_origin_map, i,
                          cur_partition_nums_, part_var_full_shape));
        }
      }
    }
  }
  return Status::OK();
}

Status ElasticTrainingPass::ScalingUpBackWardGraph(
    VarType var_type, Graph* g,
    std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
    const std::string& primary_variable_name,
    const std::vector<std::string>& opt_ev_names, Node* elastic_node,
    Node* p_dynamic_stitch_node, std::vector<Node*>& no_op_vec,
    int part_var_full_shape, int ev_partition_num) {
  Status s;

  switch (var_type) {
    case VarType::EMBEDDING_VAR:
    case VarType::REF_VAR:
    case VarType::RESOURCE_VAR:
      // corresponding to tensorflow.python.training.optimizer.py :
      // _resource_apply_sparse_duplicate_indices
      s = ScalingUpEmbeddingVariableBackWardGraph(
          var_type, g, node_to_origin_map, primary_variable_name, opt_ev_names,
          elastic_node, p_dynamic_stitch_node, no_op_vec, part_var_full_shape,
          ev_partition_num);
      break;
    default:
      s = ScalingUpDenseBackWardGraph(
          var_type, g, node_to_origin_map, primary_variable_name, opt_ev_names,
          elastic_node, no_op_vec, part_var_full_shape, ev_partition_num);
      break;
  }
  return s;
}

Status ElasticTrainingPass::ScalingDownEmbeddingVariableBackWardGraph(
    VarType var_type, Graph* g,
    std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
    std::unordered_set<Node*>& nodes_to_delete,
    const std::string& primary_ev_name,
    const std::vector<std::string>& opt_ev_names, Node* elastic_node,
    Node* p_dynamic_stitch_node, int part_var_full_shape,
    int ev_partition_num) {
  for (int i = 0; i < ev_partition_num; ++i) {
    Node* cur_ev_node = node_to_origin_map[primary_ev_name][i];
    if (i < cur_partition_nums_) {
      TF_RETURN_IF_ERROR(UpdateOldBackWardGraph(var_type, cur_ev_node,
                                                elastic_node, opt_ev_names, g,
                                                i, cur_partition_nums_));
    } else {
      TF_RETURN_IF_ERROR(DeleteSparseBackWardGraph(
          var_type, cur_ev_node, opt_ev_names, nodes_to_delete));
    }
  }

  return Status::OK();
}

Status ElasticTrainingPass::ScalingDownDenseBackWardGraph(
    VarType var_type, Graph* g,
    std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
    std::unordered_set<Node*>& nodes_to_delete,
    const std::string& primary_ev_name, Node* elastic_node,
    int part_var_full_shape, int ev_partition_num) {
  for (int i = 0; i < ev_partition_num; ++i) {
    Node* cur_var_node = node_to_origin_map[primary_ev_name][i];
    if (i < cur_partition_nums_) {
      TF_RETURN_IF_ERROR(UpdateOldDenseBackWardGraph(var_type, g, cur_var_node,
                                                     part_var_full_shape, i,
                                                     cur_partition_nums_));
    } else {
      TF_RETURN_IF_ERROR(DeleteDenseBackWardGraph(
          var_type, cur_var_node, cur_partition_nums_, nodes_to_delete));
    }
  }

  return Status::OK();
}

Status ElasticTrainingPass::ScalingDownBackWardGraph(
    Graph* g, VarType var_type,
    std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
    std::unordered_set<Node*>& nodes_to_delete,
    const std::string& primary_variable_name,
    const std::vector<std::string>& opt_ev_names, Node* elastic_node,
    Node* p_dynamic_stitch_node, int part_var_full_shape,
    int ev_partition_num) {
  Status s;
  switch (var_type) {
    case VarType::EMBEDDING_VAR:
    case VarType::REF_VAR:
    case VarType::RESOURCE_VAR:
      s = ScalingDownEmbeddingVariableBackWardGraph(
          var_type, g, node_to_origin_map, nodes_to_delete,
          primary_variable_name, opt_ev_names, elastic_node,
          p_dynamic_stitch_node, part_var_full_shape, ev_partition_num);
      break;
    default:
      s = ScalingDownDenseBackWardGraph(var_type, g, node_to_origin_map,
                                        nodes_to_delete, primary_variable_name,
                                        elastic_node, part_var_full_shape,
                                        ev_partition_num);
      break;
  }
  return s;
}

Status ElasticTrainingPass::RewriteElasticPartitionGraph(
    VarType var_type, Graph* g, std::vector<Node*>& ev_node_vec,
    Node** elastic_node, Node** p_dynamic_stitch_node,
    std::unordered_set<Node*>& nodes_to_delete) {
  Node* dynamic_partition_node = nullptr;
  Node* dynamic_stitch_node = nullptr;
  std::vector<Node*> identity_node_vec;
  std::vector<Node*> gather_node_vec;
  TF_RETURN_IF_ERROR(InitDynamicPartitionGraphMeta(
      var_type, cur_partition_nums_, ev_node_vec, gather_node_vec,
      identity_node_vec, &dynamic_partition_node, &dynamic_stitch_node));

  TF_RETURN_IF_ERROR(MakeElasticPartitionOp(
      dynamic_partition_node, cur_partition_nums_, gather_node_vec, g,
      elastic_node, nodes_to_delete));

  MakeDynamicStitchOp(dynamic_stitch_node, cur_partition_nums_,
                      identity_node_vec, g, *elastic_node,
                      p_dynamic_stitch_node, nodes_to_delete);

  return Status::OK();
}

Status ElasticTrainingPass::ScalingUpEVRedistributionGraph(
    VarType var_type, Graph* g, std::vector<Node*>& ev_node_vec,
    Node* import_op_main, int ev_partition_num,
    std::vector<Node*>& primary_ev_filters) {
  Status s;
  DataType key_type, value_type;
  std::vector<Node*> filtered_node_vec;
  filtered_node_vec.reserve(cur_partition_nums_);
  for (int i = 0; i < cur_partition_nums_; ++i) {
    auto* ev_node = ev_node_vec[i];
    auto* primary_ev_filter_node = primary_ev_filters[i];
    if (i < ev_partition_num) {
      TF_RETURN_IF_ERROR(FindNodeAndExec(
          ev_node, kEvExportOp,
          [this, &g, &filtered_node_vec, &primary_ev_filters,
           &primary_ev_filter_node, &key_type, &value_type,
           i](Node* target_node) {
            TF_RETURN_IF_ERROR(
                GetNodeAttr(target_node->attrs(), "Tkeys", &key_type));
            TF_RETURN_IF_ERROR(
                GetNodeAttr(target_node->attrs(), "dtype", &value_type));
            filtered_node_vec.push_back(target_node);
            if (primary_ev_filter_node == nullptr) {
              primary_ev_filters[i] = target_node;
            } else {
              g->AddControlEdge(target_node, primary_ev_filters[i]);
            }
            return Status::OK();
          }));
    } else {
      NodeDef filter_storage_node_def;
      TF_RETURN_IF_ERROR(
          NodeDefBuilder("FilterStorage/" + ev_node->name(), kEvExportOp)
              .Input(ev_node->name(), 0, ev_node->output_type(0))
              .Input("partition_num", 0, DT_INT32)
              .Attr("partition_id", i)
              .Attr("Tkeys", key_type)
              .Attr("dtype", value_type)
              .Finalize(&filter_storage_node_def));
      Node* filter_node = g->AddNode(filter_storage_node_def, &s);
      TF_RETURN_IF_ERROR(s);
      filter_node->set_assigned_device_name(ev_node->assigned_device_name());
      filtered_node_vec.push_back(filter_node);
      if (primary_ev_filter_node == nullptr) {
        primary_ev_filters[i] = filter_node;
      } else {
        g->AddControlEdge(filter_node, primary_ev_filters[i]);
      }
    }
  }

  for (int i = 0; i < cur_partition_nums_; ++i) {
    auto* ev_node = ev_node_vec[i];
    std::vector<Node*> sorted_filted_vec{filtered_node_vec[i]};
    for (int j = 0; j < filtered_node_vec.size(); ++j) {
      if (i != j) {
        sorted_filted_vec.emplace_back(filtered_node_vec[j]);
      }
    }

    if (i < ev_partition_num) {
      for (auto* o_node : ev_node->out_nodes()) {
        if (o_node->type_string() == kEvImportOp) {
          o_node->ClearAttr("partition_nums");
          o_node->AddAttr("partition_nums", cur_partition_nums_);
          o_node->set_assigned_device_name(ev_node->assigned_device_name());
          std::vector<const Edge*> in_edges;
          in_edges.reserve(o_node->in_edges().size());
          for (auto* o_edge : o_node->in_edges()) {
            in_edges.emplace_back(o_edge);
          }
          for (const Edge* e : in_edges) {
            g->RemoveEdge(e);
          }
          for (int j = 0; j < sorted_filted_vec.size(); ++j) {
            g->AddEdge(sorted_filted_vec[j], 0, o_node, 1 + j);
            g->AddEdge(sorted_filted_vec[j], 1, o_node,
                       1 + cur_partition_nums_ + j);
            g->AddEdge(sorted_filted_vec[j], 2, o_node,
                       1 + cur_partition_nums_ * 2 + j);
            g->AddEdge(sorted_filted_vec[j], 3, o_node,
                       1 + cur_partition_nums_ * 3 + j);
          }
        }
      }
    } else {
      std::vector<NodeDefBuilder::NodeOut> import_keys;
      std::vector<NodeDefBuilder::NodeOut> import_values;
      std::vector<NodeDefBuilder::NodeOut> import_versions;
      std::vector<NodeDefBuilder::NodeOut> import_freqs;
      for (int j = 0; j < sorted_filted_vec.size(); ++j) {
        import_keys.emplace_back(sorted_filted_vec[j]->name(), 0,
                                 sorted_filted_vec[j]->output_type(0));
        import_values.emplace_back(sorted_filted_vec[j]->name(), 1,
                                   sorted_filted_vec[j]->output_type(1));
        import_versions.emplace_back(sorted_filted_vec[j]->name(), 2,
                                     sorted_filted_vec[j]->output_type(2));
        import_freqs.emplace_back(sorted_filted_vec[j]->name(), 3,
                                  sorted_filted_vec[j]->output_type(3));
      }
      NodeDef import_storage_node_def;
      TF_RETURN_IF_ERROR(
          NodeDefBuilder(kEvImportOp + ev_node->name(), kEvImportOp)
              .Input(ev_node->name(), 0, ev_node->output_type(0))
              .Input(import_keys)
              .Input(import_values)
              .Input(import_versions)
              .Input(import_freqs)
              .Attr("partition_id", i)
              .Attr("partition_nums", cur_partition_nums_)
              .Attr("Tkeys", key_type)
              .Attr("dtype", value_type)
              .Finalize(&import_storage_node_def));
      Node* import_node = g->AddNode(import_storage_node_def, &s);
      TF_RETURN_IF_ERROR(s);
      import_node->set_assigned_device_name(ev_node->assigned_device_name());

      g->AddControlEdge(import_node, import_op_main);
      for (int k = 0; k < ev_partition_num; ++k) {
        auto* tmp_ev_node = ev_node_vec[k];
        for (auto* n : tmp_ev_node->out_nodes()) {
          if (n->type_string() == kEvImportOp) {
            g->AddControlEdge(import_node, n);
          }
        }
      }
    }
  }

  return s;
}

Status ElasticTrainingPass::ScalingUpResVarRedistributionGraph(
    VarType var_type, Graph* g, std::vector<Node*>& var_node_vec,
    Node* import_op_main, int ev_partition_num,
    std::vector<Node*>& primary_ev_filters) {
  Status s;
  Node* ori_var = var_node_vec[0];
  Node* rhs_value_node = nullptr;
  Node* partition_num_node = nullptr;
  int partition_num;
  DataType key_type;
  for (auto* oo_node : ori_var->out_nodes()) {
    if (oo_node->type_string() == "ReAssignResource") {
      TF_RETURN_IF_ERROR(
          GetNodeAttr(oo_node->attrs(), "partition_nums", &partition_num));
      TF_RETURN_IF_ERROR(oo_node->input_node(1, &rhs_value_node));
      TF_RETURN_IF_ERROR(oo_node->input_node(2, &partition_num_node));
      TF_RETURN_IF_ERROR(GetNodeAttr(oo_node->attrs(), "T", &key_type));
    }
  }

  for (int i = ev_partition_num; i < cur_partition_nums_; ++i) {
    auto* var_node = var_node_vec[i];
    NodeDef reassign_node_def;
    TF_RETURN_IF_ERROR(
        NodeDefBuilder(var_node->name() + "/ReAssignResource",
                       "ReAssignResource")
            .Input(var_node->name(), 0, DT_RESOURCE)
            .Input(rhs_value_node->name(), 0, rhs_value_node->output_type(0))
            .Input(partition_num_node->name(), 0,
                   partition_num_node->output_type(0))
            .Attr("partition_id", i)
            .Attr("partition_nums", partition_num)
            .Attr("T", key_type)
            .Finalize(&reassign_node_def));
    Node* reassign_node = g->AddNode(reassign_node_def, &s);
    TF_RETURN_IF_ERROR(s);
    reassign_node->set_assigned_device_name(var_node->assigned_device_name());
    g->AddControlEdge(reassign_node, import_op_main);
  }
  return s;
}

Status ElasticTrainingPass::ScalingUpVarRedistributionGraph(
    VarType var_type, Graph* g, std::vector<Node*>& var_node_vec,
    Node* import_op_main, int ev_partition_num,
    std::vector<Node*>& primary_ev_filters) {
  Status s;
  Node* ori_var = var_node_vec[0];
  bool use_locking;
  int partition_num;
  DataType key_type;
  Node* rhs_value_node = nullptr;
  Node* partition_num_node = nullptr;
  for (auto* oo_node : ori_var->out_nodes()) {
    if (oo_node->type_string() == "ReAssign") {
      TF_RETURN_IF_ERROR(
          GetNodeAttr(oo_node->attrs(), "use_locking", &use_locking));
      TF_RETURN_IF_ERROR(
          GetNodeAttr(oo_node->attrs(), "partition_nums", &partition_num));
      TF_RETURN_IF_ERROR(GetNodeAttr(oo_node->attrs(), "T", &key_type));
      TF_RETURN_IF_ERROR(oo_node->input_node(1, &rhs_value_node));
      TF_RETURN_IF_ERROR(oo_node->input_node(2, &partition_num_node));
    }
  }

  for (int i = ev_partition_num; i < cur_partition_nums_; ++i) {
    auto* var_node = var_node_vec[i];
    NodeDef reassign_node_def;
    TF_RETURN_IF_ERROR(
        NodeDefBuilder(var_node->name() + "/ReAssign", "ReAssign")
            .Input(var_node->name(), 0, MakeRefType(key_type))
            .Input(rhs_value_node->name(), 0, rhs_value_node->output_type(0))
            .Input(partition_num_node->name(), 0,
                   partition_num_node->output_type(0))
            .Attr("use_locking", use_locking)
            .Attr("partition_id", i)
            .Attr("partition_nums", partition_num)
            .Attr("T", key_type)
            .Finalize(&reassign_node_def));
    Node* reassign_node = g->AddNode(reassign_node_def, &s);
    TF_RETURN_IF_ERROR(s);
    reassign_node->set_assigned_device_name(var_node->assigned_device_name());
    g->AddControlEdge(reassign_node, import_op_main);
  }
  return s;
}

Status ElasticTrainingPass::ScalingUpRedistributionGraph(
    VarType var_type, Graph* g, std::vector<Node*>& var_vec,
    Node* import_op_main, int ev_partition_num,
    std::vector<Node*>& primary_ev_filters) {
  Status s;
  switch (var_type) {
    case VarType::EMBEDDING_VAR:
      s = ScalingUpEVRedistributionGraph(var_type, g, var_vec, import_op_main,
                                         ev_partition_num, primary_ev_filters);
      break;
    case VarType::RESOURCE_VAR:
    case VarType::DENSE_RESOUCE_VAR:
      s = ScalingUpResVarRedistributionGraph(var_type, g, var_vec,
                                             import_op_main, ev_partition_num,
                                             primary_ev_filters);
      break;
    default:
      s = ScalingUpVarRedistributionGraph(var_type, g, var_vec, import_op_main,
                                          ev_partition_num, primary_ev_filters);
      break;
  }
  return s;
}

Status ElasticTrainingPass::ScalingDownRedistributionGraph(
    VarType& var_type, Graph* g, std::vector<Node*>& ev_node_vec,
    std::unordered_set<Node*>& nodes_to_delete, int ev_partition_num) {
  Status s;
  switch (var_type) {
    case VarType::EMBEDDING_VAR:
      s = ScalingDownEVRedistributionGraph(g, ev_node_vec, nodes_to_delete,
                                           ev_partition_num);
      break;
    case VarType::RESOURCE_VAR:
      s = ScalingDownResVarRedistributionGraph(g, ev_node_vec, nodes_to_delete,
                                               ev_partition_num);
      break;
    default:
      s = ScalingDownVarRedistributionGraph(g, ev_node_vec, nodes_to_delete,
                                            ev_partition_num);
      break;
  }
  return s;
}

Status ElasticTrainingPass::ScalingDownEVRedistributionGraph(
    Graph* g, std::vector<Node*>& ev_node_vec,
    std::unordered_set<Node*>& nodes_to_delete, int ev_partition_num) {
  Status s;
  DataType key_type, value_type;
  std::vector<Node*> filtered_node_vec;
  filtered_node_vec.reserve(cur_partition_nums_);

  for (int i = 0; i < ev_partition_num; ++i) {
    auto* ev_node = ev_node_vec[i];
    for (auto* o_node : ev_node->out_nodes()) {
      if (o_node->type_string() == kEvExportOp) {
        TF_RETURN_IF_ERROR(GetNodeAttr(o_node->attrs(), "Tkeys", &key_type));
        TF_RETURN_IF_ERROR(GetNodeAttr(o_node->attrs(), "dtype", &value_type));
        if (i < cur_partition_nums_) {
          filtered_node_vec.push_back(o_node);
        } else {
          nodes_to_delete.insert(o_node);
        }
      }
    }
  }

  for (int i = 0; i < ev_partition_num; ++i) {
    auto* ev_node = ev_node_vec[i];
    std::vector<Node*> sorted_filted_vec{filtered_node_vec[i]};
    for (int j = 0; j < filtered_node_vec.size(); ++j) {
      if (i != j) {
        sorted_filted_vec.emplace_back(filtered_node_vec[j]);
      }
    }
    for (auto* o_node : ev_node->out_nodes()) {
      if (o_node->type_string() == kEvImportOp) {
        if (i < cur_partition_nums_) {
          string import_op_name = o_node->name();
          nodes_to_delete.insert(o_node);
          std::vector<NodeDefBuilder::NodeOut> import_keys;
          std::vector<NodeDefBuilder::NodeOut> import_values;
          std::vector<NodeDefBuilder::NodeOut> import_versions;
          std::vector<NodeDefBuilder::NodeOut> import_freqs;
          for (int j = 0; j < sorted_filted_vec.size(); ++j) {
            import_keys.emplace_back(filtered_node_vec[j]->name(), 0,
                                     filtered_node_vec[j]->output_type(0));
            import_values.emplace_back(filtered_node_vec[j]->name(), 1,
                                       filtered_node_vec[j]->output_type(1));
            import_versions.emplace_back(filtered_node_vec[j]->name(), 2,
                                         filtered_node_vec[j]->output_type(2));
            import_freqs.emplace_back(filtered_node_vec[j]->name(), 3,
                                      filtered_node_vec[j]->output_type(3));
          }
          NodeDef import_storage_node_def;
          TF_RETURN_IF_ERROR(
              NodeDefBuilder(import_op_name + "/Import", kEvImportOp)
                  .Input(ev_node->name(), 0, ev_node->output_type(0))
                  .Input(import_keys)
                  .Input(import_values)
                  .Input(import_versions)
                  .Input(import_freqs)
                  .Attr("partition_id", i)
                  .Attr("partition_nums", cur_partition_nums_)
                  .Attr("Tkeys", key_type)
                  .Attr("dtype", value_type)
                  .Finalize(&import_storage_node_def));
          Node* import_node = g->AddNode(import_storage_node_def, &s);
          TF_RETURN_IF_ERROR(s);
          import_node->set_assigned_device_name(
              ev_node->assigned_device_name());
        } else {
          nodes_to_delete.insert(o_node);
        }
      }
    }
  }

  return s;
}

Status ElasticTrainingPass::ScalingDownResVarRedistributionGraph(
    Graph* g, std::vector<Node*>& ev_node_vec,
    std::unordered_set<Node*>& nodes_to_delete, int ev_partition_num) {
  for (int i = cur_partition_nums_; i < ev_node_vec.size(); ++i) {
    auto* ev_node = ev_node_vec[i];
    for (auto* o_node : ev_node->out_nodes()) {
      if (o_node->type_string() == "ReAssignResource") {
        nodes_to_delete.emplace(o_node);
      }
    }
  }
  return Status::OK();
}

Status ElasticTrainingPass::ScalingDownVarRedistributionGraph(
    Graph* g, std::vector<Node*>& ev_node_vec,
    std::unordered_set<Node*>& nodes_to_delete, int ev_partition_num) {
  for (int i = 0; i < ev_partition_num; ++i) {
    auto* ev_node = ev_node_vec[i];
    if (i < cur_partition_nums_) {
      for (auto* o_node : ev_node->out_nodes()) {
        if (o_node->type_string() == kIdentityOp) {
          for (auto* o_edge : o_node->out_edges()) {
            // Normal Variable
            if (o_edge->dst()->type_string() == "ConcatV2") {
              int N;
              TF_RETURN_IF_ERROR(GetNodeAttr(o_edge->dst()->attrs(), "N", &N));
              if (N != cur_partition_nums_) {
                const Edge* axis_edge = nullptr;
                TF_RETURN_IF_ERROR(o_edge->dst()->input_edge(N, &axis_edge));
                g->AddEdge(axis_edge->src(), 0, o_edge->dst(),
                           cur_partition_nums_);
                g->RemoveEdge(axis_edge);
                o_edge->dst()->ClearAttr("N");
                o_edge->dst()->AddAttr("N", cur_partition_nums_);
              }
              if (o_edge->dst_input() != i) {
                g->AddEdge(o_node, 0, o_edge->dst(), i);
                g->RemoveEdge(o_edge);
              }
            }
          }
        }
      }
    } else {
      for (auto* o_node : ev_node->out_nodes()) {
        if (o_node->type_string() == kIdentityOp) {
          for (auto* oo_node : o_node->out_nodes()) {
            if (oo_node->type_string() == "ReAssign") {
              nodes_to_delete.emplace(oo_node);
            }
          }
        }
      }
    }
  }

  return Status::OK();
}

Status ElasticTrainingPass::InitHookMetaNode(Graph* g,
                                             ElasticHookMetaNode& meta_node) {
  Node* dataset_init = nullptr;
  for (auto* node : g->nodes()) {
    if (node->name() == kElasticSubGraphImport) {
      meta_node.m_import_op_main = node;
    } else if (node->name() == kElasticSubGraphInit) {
      meta_node.m_init_op_main = node;
    } else if (node->name() == kDatasetInit) {
      dataset_init = node;
    }
  }

  if ((dataset_init != nullptr) && (meta_node.m_init_op_main != nullptr)) {
    g->AddControlEdge(dataset_init, meta_node.m_init_op_main);
  }

  Status s;
  for (int i = 0; i < cur_partition_nums_; ++i) {
    string new_device_name =
        "/job:ps/replica:0/task:" + std::to_string(i) + "/device:CPU:0";
    NodeDef initop_def;
    TF_RETURN_IF_ERROR(
        NodeDefBuilder("new_sub_graph/InitOp_" + std::to_string(i), "NoOp")
            .Finalize(&initop_def));
    Node* init_node = g->AddNode(initop_def, &s);
    init_node->set_assigned_device_name(new_device_name);
    TF_RETURN_IF_ERROR(s);
    meta_node.m_init_op_vec[i] = init_node;
    g->AddControlEdge(meta_node.m_init_op_vec[i], meta_node.m_init_op_main);
  }

  NodeDef initop_def;
  TF_RETURN_IF_ERROR(NodeDefBuilder("new_sub_graph/tmp_value/InitOp", "NoOp")
                         .Finalize(&initop_def));
  meta_node.m_tmp_value_init_op = g->AddNode(initop_def, &s);
  TF_RETURN_IF_ERROR(s);
  g->AddControlEdge(meta_node.m_tmp_value_init_op, meta_node.m_init_op_main);
  return s;
}

Status ElasticTrainingPass::InitVarMeta(
    Graph* g,
    std::unordered_map<std::string, PartitionVarMeta>& primary_node_metas_map,
    std::unordered_map<std::string, std::vector<std::string>>&
        primary_node_to_opt_map,
    std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
    std::unordered_map<std::string, Node*>& unpartitioned_node_map) {
  std::unordered_map<std::string, std::vector<int>> partitioned_node_idx_map;
  for (auto* node : g->op_nodes()) {
    string node_name = node->name();
    int device_idx = 0;  // partition_num == 1 is placed on ps_0 by default
    int partiton_size = 0;
    VarType var_type;
    bool partitioned = false;
    if (node->IsKvVarHandle()) {
      var_type = VarType::EMBEDDING_VAR;
      TF_RETURN_IF_ERROR(FindNodeAndExec(node, kEvExportOp,
          [this, &device_idx](Node* target_node) {
            TF_RETURN_IF_ERROR(
              GetNodeAttr(target_node->attrs(), "partition_id", &device_idx));
            return Status::OK();
          }));
      TF_RETURN_IF_ERROR(FindNodeAndExec(node, kEvImportOp,
          [this, &partitioned](Node* target_node) {
            int partition_num;
            TF_RETURN_IF_ERROR(
              GetNodeAttr(target_node->attrs(), "partition_nums", &partition_num));
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
            node, "ReAssign",
            [this, &device_idx, &partitioned](Node* target_node) {
              TF_RETURN_IF_ERROR(GetNodeAttr(target_node->attrs(),
                                             "partition_id", &device_idx));
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
          node, "ReAssignResource",
          [this, &device_idx, &partitioned](Node* target_node) {
            TF_RETURN_IF_ERROR(
                GetNodeAttr(target_node->attrs(), "partition_id", &device_idx));
            partitioned = true;
            return Status::OK();
          }));
    } else {
      continue;
    }

    if (partitioned) {
      auto part_idx = node_name.find(kPart);
      std::string pre_str = node_name.substr(0, part_idx - 1);
      std::string post_str = node_name.substr(part_idx + strlen(kPart));
      auto post_idx = post_str.find("/");
      if (post_idx == string::npos) {
        if (primary_node_metas_map.find(pre_str) ==
            primary_node_metas_map.end()) {
          PartitionVarMeta var_meta;
          var_meta.m_var_type = var_type;
          var_meta.m_full_shape = partiton_size;
          var_meta.m_partition_num = 1;
          primary_node_metas_map.emplace(pre_str, std::move(var_meta));
          partitioned_node_idx_map.emplace(pre_str, std::vector<int>{device_idx});
          node_to_origin_map.emplace(pre_str, std::vector<Node*>{node});
        } else {
          primary_node_metas_map[pre_str].m_full_shape += partiton_size;
          primary_node_metas_map[pre_str].m_partition_num++;
          partitioned_node_idx_map[pre_str].emplace_back(device_idx);
          node_to_origin_map[pre_str].emplace_back(node);
        }
        // exactly once
        if (device_idx == 0) {
          if (primary_node_to_opt_map.find(pre_str) ==
              primary_node_to_opt_map.end()) {
            primary_node_to_opt_map.emplace(pre_str, std::vector<string>{});
          }
        }
      } else {
        string opt_name = pre_str + post_str.substr(post_idx);
        if (primary_node_metas_map.find(opt_name) ==
            primary_node_metas_map.end()) {
          PartitionVarMeta var_meta;
          var_meta.m_var_type = var_type;
          var_meta.m_full_shape = partiton_size;
          var_meta.m_partition_num = 1;
          primary_node_metas_map.emplace(opt_name, std::move(var_meta));
          partitioned_node_idx_map.emplace(opt_name, std::vector<int>{device_idx});
          node_to_origin_map.emplace(opt_name, std::vector<Node*>{node});
        } else {
          primary_node_metas_map[opt_name].m_full_shape += partiton_size;
          primary_node_metas_map[opt_name].m_partition_num++;
          partitioned_node_idx_map[opt_name].emplace_back(device_idx);
          node_to_origin_map[opt_name].emplace_back(node);
        }
        // exactly once
        if (device_idx == 0) {
          auto sep_idx = opt_name.rfind("/");
          string primary_ev_name = opt_name.substr(0, sep_idx);
          if (primary_node_to_opt_map.find(pre_str) ==
              primary_node_to_opt_map.end()) {
            primary_node_to_opt_map.emplace(pre_str,
                                            std::vector<string>{opt_name});
          } else {
            primary_node_to_opt_map[pre_str].emplace_back(opt_name);
          }
        }
      }
    } else {
      if (node_name.find(kElasticImportScope) == string::npos) {
        unpartitioned_node_map.emplace(node_name, node);
      }
    }
  }
  for (auto &it: partitioned_node_idx_map) {
    auto& old_vec = node_to_origin_map[it.first];
    int size = old_vec.size();
    std::vector<Node*> new_vec(size, nullptr);
    for (int i = 0; i < size; ++i) {
      new_vec[i] = old_vec[it.second[i]];
    }
    old_vec.swap(new_vec);
  }
  return Status::OK();
}

REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_PLACEMENT, 0,
                      ElasticTrainingPass);

}  // namespace tensorflow
