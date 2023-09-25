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

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

class CacheCKPTPass : public GraphOptimizationPass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override {
    if (!IsEnableCacheCKPT()) {
      return Status::OK();
    }

    VLOG(2) << "CacheCKPTPass: Enable Cache CKPT.";
    cache_ckpt_path_ = GetLocalCacheCKPTPath();
    cache_ckpt_replica_ = GetCacheCKPTReplica();

    Graph* graph = options.graph->get();
    if (graph == nullptr) {
      return errors::Internal("a graph should be available.");
    }
    std::unique_ptr<Graph> new_graph(new Graph(OpRegistry::Global()));
    CopyGraph(*graph, new_graph.get());

    struct RelativeNodes relative_nodes;
    bool find_relative_nodes = GetRelativeNodes(new_graph, relative_nodes);
    if (!find_relative_nodes) {
      VLOG(2) << "CacheCKPTPass: Failed to get relative nodes, skip this pass.";
      return Status::OK();
    }

    // Get num_shards.
    int num_shards = 0;
    Node* sharded_filename_op = *(relative_nodes.sharded_filename_ops.begin());
    Node* num_shards_op = nullptr;
    Tensor num_shards_tensor;
    TF_RETURN_IF_ERROR(sharded_filename_op->input_node(2, &num_shards_op));
    TF_RETURN_IF_ERROR(GetNodeAttr(num_shards_op->attrs(), "value",
                                   &num_shards_tensor));
    num_shards = num_shards_tensor.scalar<int32>()();
    CHECK_EQ(relative_nodes.sharded_filename_ops.size(), num_shards);

    if (cache_ckpt_replica_ >= num_shards) {
      VLOG(2) << "CacheCKPTPass: environment variable 'CACHE_CKPT_REPLICA' (val: "
              << cache_ckpt_replica_
              << ") should be < num_shards (val: " << num_shards
              << "), skip this pass.";
      return Status::OK();
    }

    TF_RETURN_IF_ERROR(AddSaveCacheCKPTSubGraph(new_graph, relative_nodes,
                                                num_shards));

    options.graph->swap(new_graph);

    VLOG(2) << "CacheCKPTPass: Enable Cache CKPT Success.";
    return Status::OK();
  }

 private:
  struct RelativeNodes {
    std::unordered_set<Node*> sharded_filename_ops;
    // RestoreOp includes 'RestoreV2' and 'KvResourceImportV2'
    std::unordered_set<Node*> restore_ops;
    Node* merge_checkpoints_op;
  };

  bool GetRelativeNodes(std::unique_ptr<Graph>& graph,
                        struct RelativeNodes& relative_nodes) {
    relative_nodes.merge_checkpoints_op = nullptr;
    bool find_merge_ckpt_op = false;
    for (Node* n : graph->op_nodes()) {
      if (n->type_string() == "ShardedFilename") {
        relative_nodes.sharded_filename_ops.insert(n);
      } else if (n->type_string() == "RestoreV2" ||
                 n->type_string() == "KvResourceImportV2") {
        relative_nodes.restore_ops.insert(n);
      } else if (n->type_string() == "MergeV2Checkpoints") {
        if (find_merge_ckpt_op) {
          VLOG(2) << "CacheCKPTPass: The graph contains multiple MergeV2Checkpoints nodes.";
          return false;
        }
        relative_nodes.merge_checkpoints_op = n;
        find_merge_ckpt_op = true;
      }
    }

    auto& sharded_ops = relative_nodes.sharded_filename_ops;
    auto& restore_ops = relative_nodes.restore_ops;
    if (sharded_ops.size() < 2) {
      VLOG(2) << "CacheCKPTPass: Failed to find ShardedFilename op, the num is "
              << sharded_ops.size() << ", which should be >= 2.";
      return false;
    } else if (restore_ops.size() < 2) {
      VLOG(2) << "CacheCKPTPass: Failed to find RestoreV2 or KvResourceImportV2, the num is "
              << restore_ops.size() << ", which should be >= 2.";
      return false;
    } else if (!find_merge_ckpt_op) {
      VLOG(2) << "CacheCKPTPass: Failed to find MergeV2Checkpoints.";
    }

    return true;
  }

  Status AddSaveCacheCKPTSubGraph(std::unique_ptr<Graph>& graph,
                                  const struct RelativeNodes& relative_node,
                                  int num_shards) {
    // Get ckpt_path_prefix node.
    Node* merge_checkpoints_op = relative_node.merge_checkpoints_op;
    Node* ckpt_path_prefix = nullptr;
    TF_RETURN_IF_ERROR(merge_checkpoints_op->input_node(1, &ckpt_path_prefix));

    // Generate save subgraph for each ShardedFilename/Save op.
    auto& sharded_filename_ops = relative_node.sharded_filename_ops;
    std::vector<Node*> shard_id_to_sharded_filename_node;
    shard_id_to_sharded_filename_node.resize(num_shards);
    for (Node* n : sharded_filename_ops) {
      Node* shard_op = nullptr;
      Tensor shard_tensor;
      int shard;
      TF_RETURN_IF_ERROR(n->input_node(1, &shard_op));
      TF_RETURN_IF_ERROR(GetNodeAttr(shard_op->attrs(), "value",
                                     &shard_tensor));
      shard = shard_tensor.scalar<int32>()();
      CHECK_LT(shard, num_shards);
      shard_id_to_sharded_filename_node[shard] = n;
    }

    std::vector<Node*> shard_id_to_generate_cache_ckpt_node;
    std::vector<std::unordered_set<Node*>>
      shard_id_to_recv_remote_cache_ckpt_node;

    shard_id_to_generate_cache_ckpt_node.resize(num_shards);
    shard_id_to_recv_remote_cache_ckpt_node.resize(num_shards);
    for (int i = 0; i < num_shards; i++) {
      TF_RETURN_IF_ERROR(
        GenerateSaveCacheCKPTSubGraph(graph, ckpt_path_prefix,
                                      shard_id_to_sharded_filename_node,
                                      i, num_shards,
                                      shard_id_to_generate_cache_ckpt_node,
                                      shard_id_to_recv_remote_cache_ckpt_node));
    }

    // connect the control edge.
    Node* save_run_node = nullptr;
    for (const Edge* e : merge_checkpoints_op->out_edges()) {
      if (e->IsControlEdge()) {
        save_run_node = e->dst();
        break;
      }
    }
    CHECK_NE(save_run_node, nullptr);

    for (size_t i = 0; i < shard_id_to_generate_cache_ckpt_node.size(); i++) {
      Node* generate_cache_ckpt = shard_id_to_generate_cache_ckpt_node[i];
      graph->AddControlEdge(merge_checkpoints_op, generate_cache_ckpt);
      for (Node* n : shard_id_to_recv_remote_cache_ckpt_node[i]) {
        graph->AddControlEdge(generate_cache_ckpt, n);
        graph->AddControlEdge(n, save_run_node);
      }
    }

    return Status::OK();
  }

  Status GenerateSaveCacheCKPTSubGraph(
           std::unique_ptr<Graph>& g, Node* ckpt_path_prefix,
           const std::vector<Node*>& shard_id_to_shard_filename_node,
           int shard_id, int num_shards,
           std::vector<Node*>& shard_id_to_generate_cache_ckpt_node,
           std::vector<std::unordered_set<Node*>>&
             shard_id_to_recv_remote_cache_ckpt_node) {
    Node* sharded_filename_op = shard_id_to_shard_filename_node[shard_id];
    auto split_end = sharded_filename_op->name().rfind("/");
    std::string namespace_prefix = \
      sharded_filename_op->name().substr(0, split_end) +"/SaveCacheCKPT";
    if (shard_id > 0) {
      namespace_prefix += "_" + std::to_string(shard_id);
    }
    auto& device_name = sharded_filename_op->assigned_device_name();

    // Create inputs of 'GenerateCacheCKPT' op.
    Node* shard_op = nullptr;
    Node* num_shards_op = nullptr;
    Node* cache_ckpt_path_op = nullptr;
    TF_RETURN_IF_ERROR(
        CreateInputNodesForGenerateCacheCKPTOp(g, namespace_prefix, device_name,
                                               shard_id, num_shards, shard_op,
                                               num_shards_op,
                                               cache_ckpt_path_op));
    CHECK_NE(shard_op, nullptr);
    CHECK_NE(num_shards_op, nullptr);
    CHECK_NE(cache_ckpt_path_op, nullptr);

    // Create 'GenerateCacheCKPT' op.
    NodeDef generate_cache_ckpt_def;
    TF_RETURN_IF_ERROR(NodeDefBuilder(namespace_prefix+"/GenerateCacheCKPT",
                                      "GenerateCacheCKPT")
                       .Device(device_name)
                       .Input(ckpt_path_prefix->name(), 0,
                              ckpt_path_prefix->output_type(0))
                       .Input(cache_ckpt_path_op->name(), 0,
                              cache_ckpt_path_op->output_type(0))
                       .Input(shard_op->name(), 0, shard_op->output_type(0))
                       .Input(num_shards_op->name(), 0,
                              num_shards_op->output_type(0))
                       .Finalize(&generate_cache_ckpt_def));
    Status s;
    Node* generate_cache_ckpt_op = g->AddNode(generate_cache_ckpt_def, &s);
    TF_RETURN_IF_ERROR(s);
    shard_id_to_generate_cache_ckpt_node[shard_id] = generate_cache_ckpt_op;

    // Add edges for 'GenerateCacheCKPT' op.
    g->AddEdge(ckpt_path_prefix, 0, generate_cache_ckpt_op, 0);
    g->AddEdge(cache_ckpt_path_op, 0, generate_cache_ckpt_op, 1);
    g->AddEdge(shard_op, 0, generate_cache_ckpt_op, 2);
    g->AddEdge(num_shards_op, 0, generate_cache_ckpt_op, 3);

    // Create 'RecvRemoteCacheCKPT' op.
    std::vector<int> remote_ckpt_shard_ids;
    GetRemoteCacheCKPTShardId(shard_id, num_shards, remote_ckpt_shard_ids);
    for (auto id : remote_ckpt_shard_ids) {
      Node* curr_sharded_filename_op = shard_id_to_shard_filename_node[id];
      auto curr_split_end = curr_sharded_filename_op->name().rfind("/");
      std::string curr_namespace = \
        curr_sharded_filename_op->name().substr(0, curr_split_end) + \
        "/SaveCacheCKPT";

      if (id > 0) {
        curr_namespace += "_" + std::to_string(id);
      }

      NodeDef new_recv_remote_cache_ckpt_def;
      std::string op_name = curr_namespace+"/RecvRemoteCacheCKPT";
      int recv_ckpt_op_num = shard_id_to_recv_remote_cache_ckpt_node[id].size();
      if (recv_ckpt_op_num > 0) {
        op_name += "_" + std::to_string(recv_ckpt_op_num);
      }
      auto& curr_device = curr_sharded_filename_op->assigned_device_name();
      TF_RETURN_IF_ERROR(NodeDefBuilder(op_name, "RecvRemoteCacheCKPT")
                         .Device(curr_device)
                         .Input(generate_cache_ckpt_op->name(), 0,
                                generate_cache_ckpt_op->output_type(0))
                         .Input(generate_cache_ckpt_op->name(), 1,
                                generate_cache_ckpt_op->output_type(1))
                         .Finalize(&new_recv_remote_cache_ckpt_def));
      Node* new_recv_remote_cache_ckpt_op = \
        g->AddNode(new_recv_remote_cache_ckpt_def, &s);
      TF_RETURN_IF_ERROR(s);

      shard_id_to_recv_remote_cache_ckpt_node[id].insert(new_recv_remote_cache_ckpt_op);
      g->AddEdge(generate_cache_ckpt_op, 0, new_recv_remote_cache_ckpt_op, 0);
      g->AddEdge(generate_cache_ckpt_op, 1, new_recv_remote_cache_ckpt_op, 1);
    }

    return Status::OK();
  }

  Status CreateInputNodesForGenerateCacheCKPTOp(
             std::unique_ptr<Graph>& g, const std::string& namespace_prefix,
             const std::string& device_name, const int shard_id,
             const int num_shards, Node*& shard_op, Node*& num_shards_op,
             Node*& cache_ckpt_path_op) {

    NodeDef shard_def;
    Tensor shard_val(DT_INT32, TensorShape({}));
    shard_val.scalar<int32>()() = shard_id;
    TF_RETURN_IF_ERROR(NodeDefBuilder(namespace_prefix+"/shard", "Const")
                       .Device(device_name)
                       .Attr("dtype", DT_INT32)
                       .Attr("value", shard_val)
                       .Finalize(&shard_def));
    Status s;
    shard_op = g->AddNode(shard_def, &s);
    TF_RETURN_IF_ERROR(s);

    NodeDef num_shards_def;
    Tensor num_shards_val(DT_INT32, TensorShape({}));
    num_shards_val.scalar<int32>()() = num_shards;
    TF_RETURN_IF_ERROR(NodeDefBuilder(namespace_prefix+"/num_shards", "Const")
                       .Device(device_name)
                       .Attr("dtype", DT_INT32)
                       .Attr("value", num_shards_val)
                       .Finalize(&num_shards_def));
    num_shards_op = g->AddNode(num_shards_def, &s);
    TF_RETURN_IF_ERROR(s);

    NodeDef cache_ckpt_path_def;
    Tensor cache_ckpt_path_val(DT_STRING, TensorShape({}));
    cache_ckpt_path_val.scalar<tstring>()() = cache_ckpt_path_;
    TF_RETURN_IF_ERROR(NodeDefBuilder(namespace_prefix+"/cache_ckpt_path",
                                      "Const")
                       .Device(device_name)
                       .Attr("dtype", DT_STRING)
                       .Attr("value", cache_ckpt_path_val)
                       .Finalize(&cache_ckpt_path_def));
    cache_ckpt_path_op = g->AddNode(cache_ckpt_path_def, &s);

    return s;
  }

  void GetRemoteCacheCKPTShardId(int shard_id, int num_shards,
                                 std::vector<int>& remote_ckpt_shard_ids) {
    for (int i = 0; i < cache_ckpt_replica_; i++) {
      int target_shard_id = (shard_id+i+1) % num_shards;
      remote_ckpt_shard_ids.push_back(target_shard_id);
    }
  }

  //----------------------------------------------------------------------------
  bool IsEnableCacheCKPT() {
    bool is_enable_cache_ckpt=false;
    TF_CHECK_OK(ReadBoolFromEnvVar("ENABLE_CACHE_CKPT", false,
                                   &is_enable_cache_ckpt));
    return is_enable_cache_ckpt;
  }

  string GetLocalCacheCKPTPath() {
    std::string cache_ckpt_path;
    TF_CHECK_OK(ReadStringFromEnvVar("CACHE_CKPT_PATH",
                  "/tmp/tf_fault_tolerance/cache_ckpt", &cache_ckpt_path));
    return cache_ckpt_path;
  }

  int64 GetCacheCKPTReplica() {
    int64 cache_ckpt_replica = 0;
    TF_CHECK_OK(ReadInt64FromEnvVar("CACHE_CKPT_REPLICA", 1,
                                    &cache_ckpt_replica));
    return cache_ckpt_replica;
  }

  std::string cache_ckpt_path_;
  int64 cache_ckpt_replica_;
};

REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_PLACEMENT, 0,
                      CacheCKPTPass);

} // End of namespace tensorflow
