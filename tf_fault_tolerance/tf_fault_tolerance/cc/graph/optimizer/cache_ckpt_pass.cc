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

#include "tf_fault_tolerance/cc/utils/cache_ckpt/storage_type.h"

namespace tensorflow {

namespace cache_ckpt_pass {
//------------------------------------------------------------------------------
// Type define.
struct RelativeNodes {
  std::unordered_set<Node*> sharded_filename_nodes;
  // RestoreOp includes 'RestoreV2' and 'KvResourceImportV2'
  std::unordered_set<Node*> restore_nodes;
  Node* merge_checkpoints_node;
  Node* ckpt_path_prefix_node;

  RelativeNodes() {
    merge_checkpoints_node = nullptr;
    ckpt_path_prefix_node = nullptr;
  }
};

namespace utils {
  void GetRemoteCacheCKPTShardId(int shard_id, int num_shards, int replica,
                                 std::vector<int>& remote_ckpt_shard_ids) {
    for (int i = 0; i < replica; i++) {
      int target_shard_id = (shard_id+i+1) % num_shards;
      remote_ckpt_shard_ids.push_back(target_shard_id);
    }
  }
} // End of namespace utils

//------------------------------------------------------------------------------
// Helper class to add save cache ckpt subgraph.
class AddSaveCacheCKPTSubGraphHelper {
 public:
  AddSaveCacheCKPTSubGraphHelper(const struct RelativeNodes& relative_nodes,
                                 const std::string& cache_path,
                                 const int64 cache_ckpt_replica,
                                 const int num_shards,
                                 const std::string& local_storage_type,
                                 const std::string& remote_storage_type)
    : relative_nodes_(relative_nodes), cache_path_(cache_path),
      cache_ckpt_replica_(cache_ckpt_replica), num_shards_(num_shards),
      local_storage_type_(local_storage_type),
      remote_storage_type_(remote_storage_type) {}

  ~AddSaveCacheCKPTSubGraphHelper() {}

  Status Run(std::unique_ptr<Graph>& g) {
    // Get the map of 'shard_id' to 'sharded_filename_node'.
    auto& sharded_filename_nodes = relative_nodes_.sharded_filename_nodes;
    std::vector<Node*> shard_to_sharded_filename_node;
    TF_RETURN_IF_ERROR(
      GetShardToShardedFilenameNodeMap(sharded_filename_nodes,
                                       shard_to_sharded_filename_node));

    // Generate save cache ckpt subgraph for each ShardedFilename/Save op.
    // map of 'shard' to 'GenerateCacheCKPT' node.
    std::vector<Node*> shard_to_generate_node;
    shard_to_generate_node.resize(num_shards_);
    // map of 'shard' to 'BackupRemoteCacheCKPT' node.
    std::vector<std::unordered_set<Node*>> shard_to_backup_node;
    shard_to_backup_node.resize(num_shards_);
    for (int i = 0; i < num_shards_; i++) {
      TF_RETURN_IF_ERROR(GenerateGraph(g, i, shard_to_sharded_filename_node,
                           shard_to_generate_node, shard_to_backup_node));
    }

    //connect the control edge.
    Node* merge_checkpoints_node = relative_nodes_.merge_checkpoints_node;
    Node* save_run_node = nullptr;
    for (const Edge* e : merge_checkpoints_node->out_edges()) {
      if (e->IsControlEdge()) {
        save_run_node = e->dst();
        break;
      }
    }
    CHECK_NE(save_run_node, nullptr);

    for (size_t i = 0; i < shard_to_generate_node.size(); i++) {
      Node* generate_node = shard_to_generate_node[i];
      g->AddControlEdge(merge_checkpoints_node, generate_node);
      for (Node* n : shard_to_backup_node[i]) {
        g->AddControlEdge(generate_node, n);
        g->AddControlEdge(n, save_run_node);
      }
    }

    return Status::OK();
  }

 private:
  // Functions
  Status GetShardToShardedFilenameNodeMap(
           const std::unordered_set<Node*>& sharded_filename_nodes,
           std::vector<Node*>& shard_to_sharded_filename_node) {
    Node* shard_node = nullptr;
    Tensor shard_tensor;

    shard_to_sharded_filename_node.resize(num_shards_);
    for (Node* n : sharded_filename_nodes) {
      TF_RETURN_IF_ERROR(n->input_node(1, &shard_node));
      TF_RETURN_IF_ERROR(GetNodeAttr(shard_node->attrs(), "value",
                                     &shard_tensor));
      int shard = shard_tensor.scalar<int32>()();
      CHECK_LT(shard, num_shards_);
      shard_to_sharded_filename_node[shard] = n;
    }

    return Status::OK();
  }

  Status GenerateGraph(std::unique_ptr<Graph>& g, const int shard,
          const std::vector<Node*>& shard_to_filename_node,
          std::vector<Node*>& shard_to_generate_node,
          std::vector<std::unordered_set<Node*>>& shard_to_backup_node) {
    Node* sharded_filename_node = shard_to_filename_node[shard];
    auto& device_name = sharded_filename_node->assigned_device_name();

    auto split_end = sharded_filename_node->name().rfind("/");
    std::string name_prefix = \
      sharded_filename_node->name().substr(0, split_end) +"/SaveCacheCKPT";
    if (shard > 0) {
      name_prefix += "_" + std::to_string(shard);
    }

    // Create 'GenerateCacheCKPT' op.
    TF_RETURN_IF_ERROR(CreateGenerateCacheCKPTOp(g, name_prefix, device_name,
                         shard, shard_to_generate_node));

    // Create 'BackupRemoteCacheCKPT' op.
    std::vector<int> remote_shard_ids;
    remote_shard_ids.reserve(cache_ckpt_replica_);
    utils::GetRemoteCacheCKPTShardId(shard, num_shards_, cache_ckpt_replica_,
                                     remote_shard_ids);
    TF_RETURN_IF_ERROR(CreateBackupRemoteCacheCKPTOp(g,
                         shard_to_generate_node[shard], remote_shard_ids,
                         shard_to_filename_node, shard_to_backup_node));

    return Status::OK();
  }

  Status CreateGenerateCacheCKPTOp(std::unique_ptr<Graph>& g,
           const std::string& name_prefix, const std::string& device_name,
           const int shard, std::vector<Node*>& shard_to_generate_node) {
    std::string node_name = name_prefix + "/GenerateCacheCKPT";
    // Create input nodes for 'GenerateCacheCKPT' node.
    Node* cache_path_node = nullptr;
    Node* shard_node = nullptr;
    Node* num_shards_node = nullptr;
    TF_RETURN_IF_ERROR(
      CreateInputNodesForGenerateCacheCKPTNode(g, node_name, device_name, shard,
        cache_path_node, shard_node, num_shards_node));

    // Create 'GenerateCacheCKPT' node.
    Node* ckpt_prefix_node = relative_nodes_.ckpt_path_prefix_node;
    NodeDef generate_def;
    TF_RETURN_IF_ERROR(NodeDefBuilder(node_name, "GenerateCacheCKPT")
                       .Device(device_name)
                       .Input(ckpt_prefix_node->name(), 0,
                              ckpt_prefix_node->output_type(0))
                       .Input(cache_path_node->name(), 0,
                              cache_path_node->output_type(0))
                       .Input(shard_node->name(), 0, shard_node->output_type(0))
                       .Input(num_shards_node->name(), 0,
                              num_shards_node->output_type(0))
                       .Attr("ckpt_storage_type", local_storage_type_)
                       .Finalize(&generate_def));
    Status s;
    Node* generate_node = g->AddNode(generate_def, &s);
    TF_RETURN_IF_ERROR(s);
    shard_to_generate_node[shard] = generate_node;

    // Add input edges for 'GenerateCacheCKPT' node.
    g->AddEdge(ckpt_prefix_node, 0, generate_node, 0);
    g->AddEdge(cache_path_node, 0, generate_node, 1);
    g->AddEdge(shard_node, 0, generate_node, 2);
    g->AddEdge(num_shards_node, 0, generate_node, 3);

    return Status::OK();
  }

  Status CreateInputNodesForGenerateCacheCKPTNode(std::unique_ptr<Graph>& g,
           const std::string& name_prefix, const std::string& device_name,
           const int shard, Node*& cache_path_node, Node*& shard_node,
           Node*& num_shards_node) {
    Status s;

    // cache_path node.
    NodeDef cache_path_def;
    Tensor cache_path_val(DT_STRING, TensorShape({}));
    cache_path_val.scalar<tstring>()() = \
      cache_path_ + "_" + std::to_string(shard);
    TF_RETURN_IF_ERROR(NodeDefBuilder(name_prefix+"/cache_path", "Const")
                       .Device(device_name)
                       .Attr("dtype", DT_STRING)
                       .Attr("value", cache_path_val)
                       .Finalize(&cache_path_def));
    cache_path_node = g->AddNode(cache_path_def, &s);
    TF_RETURN_IF_ERROR(s);

    // shard node.
    NodeDef shard_def;
    Tensor shard_val(DT_INT32, TensorShape({}));
    shard_val.scalar<int32>()() = shard;
    TF_RETURN_IF_ERROR(NodeDefBuilder(name_prefix+"/shard", "Const")
                       .Device(device_name)
                       .Attr("dtype", DT_INT32)
                       .Attr("value", shard_val)
                       .Finalize(&shard_def));
    shard_node = g->AddNode(shard_def, &s);
    TF_RETURN_IF_ERROR(s);

    // num_shards node.
    NodeDef num_shards_def;
    Tensor num_shards_val(DT_INT32, TensorShape({}));
    num_shards_val.scalar<int32>()() = num_shards_;
    TF_RETURN_IF_ERROR(NodeDefBuilder(name_prefix+"/num_shards", "Const")
                       .Device(device_name)
                       .Attr("dtype", DT_INT32)
                       .Attr("value", num_shards_val)
                       .Finalize(&num_shards_def));
    num_shards_node = g->AddNode(num_shards_def, &s);

    return s;
  }

  Status CreateBackupRemoteCacheCKPTOp(std::unique_ptr<Graph>& g,
           Node* generate_node, const std::vector<int>& remote_shard_ids,
           const std::vector<Node*>& shard_to_filename_node,
           std::vector<std::unordered_set<Node*>>& shard_to_backup_node) {
    for (auto id : remote_shard_ids) {
      Node* filename_node = shard_to_filename_node[id];
      auto split_end = filename_node->name().rfind("/");
      std::string name_prefix = \
        filename_node->name().substr(0, split_end) + "/SaveCacheCKPT";
      if (id > 0) {
        name_prefix += "_" + std::to_string(id);
      }

      std::string node_name = name_prefix + "/BackupRemoteCacheCKPT";
      int backup_node_num = shard_to_backup_node[id].size();
      if (backup_node_num > 0) {
        node_name += "_" + std::to_string(backup_node_num);
      }
      auto& device_name = filename_node->assigned_device_name();

      // cache_path node.
      NodeDef cache_path_def;
      Tensor cache_path_val(DT_STRING, TensorShape({}));
      cache_path_val.scalar<tstring>()() = \
        cache_path_ + "_" + std::to_string(id);
      TF_RETURN_IF_ERROR(NodeDefBuilder(node_name+"/cache_path", "Const")
                         .Device(device_name)
                         .Attr("dtype", DT_STRING)
                         .Attr("value", cache_path_val)
                         .Finalize(&cache_path_def));
      Status s;
      Node* cache_path_node = g->AddNode(cache_path_def, &s);
      TF_RETURN_IF_ERROR(s);

      NodeDef backup_node_def;
      TF_RETURN_IF_ERROR(NodeDefBuilder(node_name, "BackupRemoteCacheCKPT")
                         .Device(device_name)
                         .Input(cache_path_node->name(), 0,
                                cache_path_node->output_type(0))
                         .Input(generate_node->name(), 1,
                                generate_node->output_type(0))
                         .Input(generate_node->name(), 2,
                                generate_node->output_type(1))
                         .Input(generate_node->name(), 3,
                                generate_node->output_type(2))
                         .Attr("ckpt_storage_type", remote_storage_type_)
                         .Finalize(&backup_node_def));
      Node* backup_node = g->AddNode(backup_node_def, &s);
      TF_RETURN_IF_ERROR(s);
      shard_to_backup_node[id].insert(backup_node);

      g->AddEdge(cache_path_node, 0, backup_node, 0);
      g->AddEdge(generate_node, 0, backup_node, 1);
      g->AddEdge(generate_node, 1, backup_node, 2);
      g->AddEdge(generate_node, 2, backup_node, 3);
    }

    return Status::OK();
  }

  // Variables.
  struct RelativeNodes relative_nodes_;
  std::string cache_path_;
  int64 cache_ckpt_replica_;
  int num_shards_;
  std::string local_storage_type_;
  std::string remote_storage_type_;
};

//------------------------------------------------------------------------------
// Helper class to add restore cache ckpt subgraph.
class AddRestoreCacheCKPTSubGraphHelper {
 public:
  AddRestoreCacheCKPTSubGraphHelper(const struct RelativeNodes& relative_nodes,
                                    const std::string& cache_path,
                                    const int64 cache_ckpt_replica,
                                    const int num_shards,
                                    const std::string& local_storage_type)
    : relative_nodes_(relative_nodes), cache_path_(cache_path),
      cache_ckpt_replica_(cache_ckpt_replica), num_shards_(num_shards),
      local_storage_type_(local_storage_type) {}

  ~AddRestoreCacheCKPTSubGraphHelper() {}

  Status Run(std::unique_ptr<Graph>& g) {
    auto& shard_filename_nodes = relative_nodes_.sharded_filename_nodes;

    // Get shard id to device map.
    std::vector<std::string> shard_to_device;
    TF_RETURN_IF_ERROR(
      GetShardToDeviceMap(shard_filename_nodes, shard_to_device));

    // Add restore cache subgraph.
    std::vector<int> remote_shard_ids;
    remote_shard_ids.reserve(cache_ckpt_replica_);
    std::vector<int> repatriate_counters(num_shards_, 0);
    for (int shard = 0; shard < num_shards_; shard++) {
      remote_shard_ids.clear();
      utils::GetRemoteCacheCKPTShardId(shard, num_shards_, cache_ckpt_replica_,
                                       remote_shard_ids);
      Node* restore_output_node = nullptr;
      TF_RETURN_IF_ERROR(
        AddRestoreCacheCKPTGraph(g, shard, remote_shard_ids, shard_to_device,
                                 repatriate_counters, restore_output_node));

      // Update Edges.
      if (local_storage_type_ == StorageType::kPosixFileType) {
        const std::string& device_name = shard_to_device[shard];
        for (Node* n: relative_nodes_.restore_nodes) {
          if (n->assigned_device_name() == device_name) {
            TF_RETURN_IF_ERROR(g->UpdateEdge(restore_output_node, 1, n, 0));
          }
        }
      } else {
        // TODO: Support more storage type for local cache ckpt.
        // May need to implement new Save/Restore op, such as saving ckpt in
        // memory or loading ckpt from memory.
        LOG(FATAL) << "CacheCKPTPass: local cache ckpt only support posix file now.";
      }
    }

    return Status::OK();
  }

 private:
  // Functions.
  Status GetShardToDeviceMap(const std::unordered_set<Node*> filename_nodes,
                             std::vector<std::string>& shard_to_device) {
    shard_to_device.resize(num_shards_);

    Node* shard_node = nullptr;
    Tensor shard_tensor;
    int shard = 0;
    for (Node* n : filename_nodes) {
      TF_RETURN_IF_ERROR(n->input_node(1, &shard_node));
      TF_RETURN_IF_ERROR(GetNodeAttr(shard_node->attrs(), "value",
                                     &shard_tensor));
      shard = shard_tensor.scalar<int32>()();
      const std::string& device_name = n->assigned_device_name();
      shard_to_device[shard] = device_name;
    }
    return Status::OK();
  }

  Status AddRestoreCacheCKPTGraph(std::unique_ptr<Graph>& g, int shard,
           const std::vector<int>& remote_shard_ids,
           const std::vector<std::string>& shard_to_device,
           std::vector<int>& repatriate_counters, Node*& restore_output_node) {
    Node* ckpt_path_prefix_node = relative_nodes_.ckpt_path_prefix_node;
    auto split_end = ckpt_path_prefix_node->name().rfind("/");
    std::string name_prefix = \
      ckpt_path_prefix_node->name().substr(0, split_end) + "/RestoreCacheCKPT";
    if(shard > 0) {
      name_prefix += "_" + std::to_string(shard);
    }

    const std::string& device_name = shard_to_device[shard];
    std::vector<Node*> switch_nodes;
    switch_nodes.reserve(cache_ckpt_replica_ + 2);

    // Generate CheckLocalCacheCKPT and Switch nodes.
    Node* check_local_node = nullptr;
    TF_RETURN_IF_ERROR(AddCheckLocalCacheCKPTNodes(g, name_prefix, device_name,
                         shard, check_local_node, switch_nodes));

    // Generate GetRemoteCacheCKPT, RepatriateRemoteCacheCKPT and Switch nodes.
    Node* in_node = check_local_node;
    Node* out_node = nullptr;
    for (size_t i = 0; i < remote_shard_ids.size(); i++) {
      int remote_shard = remote_shard_ids[i];

      TF_RETURN_IF_ERROR(
        AddFetchRemoteCacheCKPTGraph(g, name_prefix, i, shard_to_device, shard,
          remote_shard, repatriate_counters, in_node, out_node, switch_nodes));

      in_node = out_node;
    }

    // Generate LoadCKPTFromFilePath op.
    Node* load_file_ckpt_node = nullptr;
    TF_RETURN_IF_ERROR(AddLoadCKPTFromFilePathNodes(g, name_prefix, device_name,
                         shard, in_node, load_file_ckpt_node));

    // Generate Merge and UnPackCacheCKPTResource op.
    TF_RETURN_IF_ERROR(CreateMergeAndUnPackCacheCKPTResourceOp(g, name_prefix,
                         device_name, switch_nodes, load_file_ckpt_node,
                         restore_output_node));

    return Status::OK();
  }

  Status AddCheckLocalCacheCKPTNodes(std::unique_ptr<Graph>& g,
           const std::string& name_prefix, const std::string& device_name,
           const int shard, Node*& check_local_node,
           std::vector<Node*>& switch_nodes) {
    std::string node_name = name_prefix + "/CheckLocalCacheCKPT";
    TF_RETURN_IF_ERROR(CreateCheckLocalCacheCKPTOp(g, node_name, device_name,
                                                   shard, check_local_node));

    // Create 'Switch' op.
    NodeDef switch_def;
    TF_RETURN_IF_ERROR(NodeDefBuilder(node_name + "/Swtich", "Switch")
                       .Device(device_name)
                       .Input(check_local_node->name(), 1,
                              check_local_node->output_type(1))
                       .Input(check_local_node->name(), 0,
                              check_local_node->output_type(0))
                       .Finalize(&switch_def));
    Status s;
    Node* switch_node = g->AddNode(switch_def, &s);
    TF_RETURN_IF_ERROR(s);

    // connect the edges for Switch.
    g->AddEdge(check_local_node, 1, switch_node, 0);
    g->AddEdge(check_local_node, 0, switch_node, 1);

    switch_nodes.push_back(switch_node);

    return Status::OK();
  }

  Status CreateCheckLocalCacheCKPTOp(std::unique_ptr<Graph>& g,
                                     const std::string& node_name,
                                     const std::string& device_name,
                                     const int shard,
                                     Node*& check_local_node) {
    Node* ckpt_prefix_node = relative_nodes_.ckpt_path_prefix_node;

    // Create input nodes.
    Node* shard_node = nullptr;
    Node* num_shards_node = nullptr;
    TF_RETURN_IF_ERROR(CreateInputNodesForCheckLocalCacheCKPTOp(g, node_name,
                         device_name, shard, shard_node, num_shards_node));

    // Create 'CheckLocalCacheCKPT' op.
    NodeDef check_local_cache_ckpt_def;
    TF_RETURN_IF_ERROR(NodeDefBuilder(node_name, "CheckLocalCacheCKPT")
                       .Device(device_name)
                       .Input(ckpt_prefix_node->name(), 0,
                              ckpt_prefix_node->output_type(0))
                       .Input(shard_node->name(), 0, shard_node->output_type(0))
                       .Input(num_shards_node->name(), 0,
                              num_shards_node->output_type(0))
                       .Attr("shared_name", "cache_ckpt")
                       .Finalize(&check_local_cache_ckpt_def));
    Status s;
    check_local_node = g->AddNode(check_local_cache_ckpt_def, &s);
    TF_RETURN_IF_ERROR(s);

    // Connect edges for CheckLocalCacheCKPT.
    g->AddEdge(ckpt_prefix_node, 0, check_local_node, 0);
    g->AddEdge(shard_node, 0, check_local_node, 1);
    g->AddEdge(num_shards_node, 0, check_local_node, 2);

    return Status::OK();
  }

  Status CreateInputNodesForCheckLocalCacheCKPTOp(
           std::unique_ptr<Graph>& g, const std::string& name_prefix,
           const std::string device_name, const int shard, Node*& shard_node,
           Node*& num_shards_node) {
    NodeDef shard_def;
    Tensor shard_val(DT_INT32, TensorShape({}));
    shard_val.scalar<int32>()() = shard;
    TF_RETURN_IF_ERROR(NodeDefBuilder(name_prefix+"/shard", "Const")
                       .Device(device_name)
                       .Attr("dtype", DT_INT32)
                       .Attr("value", shard_val)
                       .Finalize(&shard_def));
    Status s;
    shard_node = g->AddNode(shard_def, &s);
    TF_RETURN_IF_ERROR(s);

    NodeDef num_shards_def;
    Tensor num_shards_val(DT_INT32, TensorShape({}));
    num_shards_val.scalar<int32>()() = num_shards_;
    TF_RETURN_IF_ERROR(NodeDefBuilder(name_prefix+"/num_shards", "Const")
                       .Device(device_name)
                       .Attr("dtype", DT_INT32)
                       .Attr("value", num_shards_val)
                       .Finalize(&num_shards_def));
    num_shards_node = g->AddNode(num_shards_def, &s);

    return s;
  }

  Status AddFetchRemoteCacheCKPTGraph(std::unique_ptr<Graph>& g,
           const std::string& name_prefix, const int index,
           const std::vector<std::string>& shard_to_device, const int shard,
           const int remote_shard, std::vector<int>& repatriate_counters,
           Node* in_check_node, Node*& get_remote_node,
           std::vector<Node*>& switch_nodes) {
    CHECK_EQ(shard_to_device.size(), num_shards_);
    CHECK_LT(shard, num_shards_);
    CHECK_LT(remote_shard, num_shards_);

    Node* ckpt_prefix_node = relative_nodes_.ckpt_path_prefix_node;
    auto split_end = ckpt_prefix_node->name().rfind("/");
    std::string remote_name_prefix = \
      ckpt_prefix_node->name().substr(0, split_end) + "/RestoreCachCKPT";
    if (remote_shard > 0) {
      remote_name_prefix += "_" + std::to_string(remote_shard);
    }

    // Generate RepatriateRemoteCacheCKPT nodes.
    Node* repatriate_node = nullptr;
    TF_RETURN_IF_ERROR(AddRepatriateRemoteCacheCKPTNodes(g, remote_name_prefix,
        repatriate_counters[remote_shard], shard_to_device[remote_shard], shard,
        in_check_node, repatriate_node));

    // Generate GetRemoteCacheCKPT nodes.
    TF_RETURN_IF_ERROR(AddGetRemoteCacheCKPTNodes(g, name_prefix, index, shard,
                         shard_to_device[shard], remote_shard, repatriate_node,
                         get_remote_node, switch_nodes));

    return Status::OK();
  }

  Status AddRepatriateRemoteCacheCKPTNodes(std::unique_ptr<Graph>& g,
           const std::string& name_prefix, int& repatriate_counter,
           const std::string& device_name, const int shard, Node* in_check_node,
           Node*& repatriate_node) {
    std::string repatriate_name = name_prefix + "/RepatriateRemoteCacheCKPT";
    if (repatriate_counter > 0) {
      repatriate_name += "_" + std::to_string(repatriate_counter);
    }
    repatriate_counter++;

    Node* ckpt_prefix_node = relative_nodes_.ckpt_path_prefix_node;
    // Create switch op.
    NodeDef switch_def;
    TF_RETURN_IF_ERROR(NodeDefBuilder(repatriate_name + "/Switch", "Switch")
                       .Device(device_name)
                       .Input(ckpt_prefix_node->name(), 0,
                              ckpt_prefix_node->output_type(0))
                       .Input(in_check_node->name(), 0,
                              in_check_node->output_type(0))
                       .Finalize(&switch_def));
    Status s;
    Node* switch_node = g->AddNode(switch_def, &s);
    TF_RETURN_IF_ERROR(s);

    // Connect input edges for Switch op.
    g->AddEdge(ckpt_prefix_node, 0, switch_node, 0);
    g->AddEdge(in_check_node, 0, switch_node, 1);

    // Create RepatriateRemoteCacheCKPT op.
    TF_RETURN_IF_ERROR(CreateRepatriateRemoteCacheCKPTOp(g, repatriate_name,
                         device_name, shard, switch_node, repatriate_node));

    return Status::OK();
  }

  Status CreateRepatriateRemoteCacheCKPTOp(std::unique_ptr<Graph>& g,
           const std::string& send_node_name, const std::string& device_name,
           const int shard, Node* switch_node, Node*& send_node) {
    // Create input nodes of RepatriateRemoteCacheCKPT.
    NodeDef shard_def;
    Tensor shard_val(DT_INT32, TensorShape({}));
    shard_val.scalar<int32>()() = shard;
    TF_RETURN_IF_ERROR(NodeDefBuilder(send_node_name+"/shard", "Const")
                       .Device(device_name)
                       .Attr("dtype", DT_INT32)
                       .Attr("value", shard_val)
                       .Finalize(&shard_def));
    Status s;
    Node* shard_node = g->AddNode(shard_def, &s);
    TF_RETURN_IF_ERROR(s);

    NodeDef num_shards_def;
    Tensor num_shards_val(DT_INT32, TensorShape({}));
    num_shards_val.scalar<int32>()() = num_shards_;
    TF_RETURN_IF_ERROR(NodeDefBuilder(send_node_name+"/num_shards", "Const")
                       .Device(device_name)
                       .Attr("dtype", DT_INT32)
                       .Attr("value", num_shards_val)
                       .Finalize(&num_shards_def));
    Node* num_shards_node = g->AddNode(num_shards_def, &s);
    TF_RETURN_IF_ERROR(s);

    // Create RepatriateRemoteCacheCKPT op.
    NodeDef send_remote_cache_ckpt_def;
    TF_RETURN_IF_ERROR(NodeDefBuilder(send_node_name,
                                      "RepatriateRemoteCacheCKPT")
                       .Device(device_name)
                       .Input(switch_node->name(), 0,
                              switch_node->output_type(0))
                       .Input(shard_node->name(), 0, shard_node->output_type(0))
                       .Input(num_shards_node->name(), 0,
                              num_shards_node->output_type(0))
                       .Finalize(&send_remote_cache_ckpt_def));
    send_node = g->AddNode(send_remote_cache_ckpt_def, &s);
    TF_RETURN_IF_ERROR(s);

    // Connect input edges for RepatriateRemoteCacheCKPT op.
    g->AddEdge(switch_node, 0, send_node, 0);
    g->AddEdge(shard_node, 0, send_node, 1);
    g->AddEdge(num_shards_node, 0, send_node, 2);

    return Status::OK();
  }

  Status AddGetRemoteCacheCKPTNodes(std::unique_ptr<Graph>& g,
           const std::string& name_prefix, const int index, const int shard,
           const std::string& device_name, const int remote_shard,
           Node* repatriate_node, Node*& get_remote_node,
           std::vector<Node*>& switch_nodes) {
    std::string get_ckpt_name = name_prefix + "/GetRemoteCacheCKPT";
    if (index > 0) {
      get_ckpt_name += "_" + std::to_string(index);
    }

    // Create GetRemoteCacheCKPT op.
    TF_RETURN_IF_ERROR(CreateGetRemoteCacheCKPTOp(g, get_ckpt_name, device_name,
                         shard, repatriate_node, get_remote_node));

    // Create Switch node.
    NodeDef switch_def;
    TF_RETURN_IF_ERROR(NodeDefBuilder(get_ckpt_name + "/Switch", "Switch")
                       .Device(device_name)
                       .Input(get_remote_node->name(), 1,
                              get_remote_node->output_type(1))
                       .Input(get_remote_node->name(), 0,
                              get_remote_node->output_type(0))
                       .Finalize(&switch_def));
    Status s;
    Node* switch_node = g->AddNode(switch_def, &s);
    TF_RETURN_IF_ERROR(s);

    // Connect input edges for switch.
    g->AddEdge(get_remote_node, 1, switch_node, 0);
    g->AddEdge(get_remote_node, 0, switch_node, 1);

    switch_nodes.push_back(switch_node);

    return Status::OK();
  }

  Status CreateGetRemoteCacheCKPTOp(std::unique_ptr<Graph>& g,
           const std::string& node_name, const std::string& device_name,
           const int shard, Node* repatriate_node, Node*& get_remote_node) {
    Status s;
    // Create inputs for GetRemoteCacheCKPT op.
    NodeDef cache_path_def;
    Tensor cache_path_val(DT_STRING, TensorShape({}));
    cache_path_val.scalar<tstring>()() = \
      cache_path_ + "_" + std::to_string(shard);
    TF_RETURN_IF_ERROR(NodeDefBuilder(node_name+"/cache_path", "Const")
                       .Device(device_name)
                       .Attr("dtype", DT_STRING)
                       .Attr("value", cache_path_val)
                       .Finalize(&cache_path_def));
    Node* cache_path_node = g->AddNode(cache_path_def, &s);
    TF_RETURN_IF_ERROR(s);

    // Create GetRemoteCacheCKPT op.
    Node* ckpt_prefix_node = relative_nodes_.ckpt_path_prefix_node;
    NodeDef get_remote_cache_ckpt_def;
    TF_RETURN_IF_ERROR(NodeDefBuilder(node_name, "GetRemoteCacheCKPT")
                       .Device(device_name)
                       .Input(ckpt_prefix_node->name(), 0,
                              ckpt_prefix_node->output_type(0))
                       .Input(cache_path_node->name(), 0,
                              cache_path_node->output_type(0))
                       .Input(repatriate_node->name(), 0,
                              repatriate_node->output_type(0))
                       .Input(repatriate_node->name(), 1,
                              repatriate_node->output_type(1))
                       .Input(repatriate_node->name(), 2,
                              repatriate_node->output_type(2))
                       .Input(repatriate_node->name(), 3,
                              repatriate_node->output_type(3))
                       .Attr("shared_name", "cache_ckpt")
                       .Attr("ckpt_storage_type", local_storage_type_)
                       .Finalize(&get_remote_cache_ckpt_def));
    get_remote_node = g->AddNode(get_remote_cache_ckpt_def, &s);
    TF_RETURN_IF_ERROR(s);

    // Connect input edges for 'CheckRemoteCacheCKPT' op.
    g->AddEdge(ckpt_prefix_node, 0, get_remote_node, 0);
    g->AddEdge(cache_path_node, 0, get_remote_node, 1);
    g->AddEdge(repatriate_node, 0, get_remote_node, 2);
    g->AddEdge(repatriate_node, 1, get_remote_node, 3);
    g->AddEdge(repatriate_node, 2, get_remote_node, 4);
    g->AddEdge(repatriate_node, 3, get_remote_node, 5);

    return Status::OK();
  }

  Status AddLoadCKPTFromFilePathNodes(std::unique_ptr<Graph>& g,
           const std::string& name_prefix, const std::string& device_name,
           const int shard, Node* check_ckpt_node, Node*& load_ckpt_node) {
    Node* shard_node = nullptr;
    Node* num_shards_node = nullptr;
    TF_RETURN_IF_ERROR(CreateInputNodesForLoadCKPTFromFilePathOp(g, name_prefix,
                         device_name, shard, shard_node, num_shards_node));
    Node* ckpt_path_prefix_node = relative_nodes_.ckpt_path_prefix_node;
    std::string load_node_name = name_prefix+"/LoadCKPTFromFilePath";

    // Create Swtich Node.
    NodeDef switch_node_def;
    TF_RETURN_IF_ERROR(NodeDefBuilder(load_node_name+"/Switch", "Switch")
                       .Device(device_name)
                       .Input(ckpt_path_prefix_node->name(), 0,
                              ckpt_path_prefix_node->output_type(0))
                       .Input(check_ckpt_node->name(), 0,
                              check_ckpt_node->output_type(0))
                       .Finalize(&switch_node_def));
    Status s;
    Node* switch_node = g->AddNode(switch_node_def, &s);
    TF_RETURN_IF_ERROR(s);
    g->AddEdge(ckpt_path_prefix_node, 0, switch_node, 0);
    g->AddEdge(check_ckpt_node, 0, switch_node, 1);

    // Create LoadCKPTFromPathFile op.
    NodeDef load_ckpt_node_def;
    TF_RETURN_IF_ERROR(NodeDefBuilder(load_node_name, "LoadCKPTFromFilePath")
                       .Device(device_name)
                       .Input(switch_node->name(), 0,
                              switch_node->output_type(0))
                       .Input(shard_node->name(), 0, shard_node->output_type(0))
                       .Input(num_shards_node->name(), 0,
                              num_shards_node->output_type(0))
                       .Finalize(&load_ckpt_node_def));
    load_ckpt_node = g->AddNode(load_ckpt_node_def, &s);
    TF_RETURN_IF_ERROR(s);
    g->AddEdge(switch_node, 0, load_ckpt_node, 0);
    g->AddEdge(shard_node, 0, load_ckpt_node, 1);
    g->AddEdge(num_shards_node, 0, load_ckpt_node, 2);


    return Status::OK();
  }

  Status CreateInputNodesForLoadCKPTFromFilePathOp(
           std::unique_ptr<Graph>& g, const std::string& name_prefix,
           const std::string device_name, const int shard, Node*& shard_node,
           Node*& num_shards_node) {
    NodeDef shard_def;
    Tensor shard_val(DT_INT32, TensorShape({}));
    shard_val.scalar<int32>()() = shard;
    TF_RETURN_IF_ERROR(NodeDefBuilder(name_prefix+"/shard", "Const")
                       .Device(device_name)
                       .Attr("dtype", DT_INT32)
                       .Attr("value", shard_val)
                       .Finalize(&shard_def));
    Status s;
    shard_node = g->AddNode(shard_def, &s);
    TF_RETURN_IF_ERROR(s);

    NodeDef num_shards_def;
    Tensor num_shards_val(DT_INT32, TensorShape({}));
    num_shards_val.scalar<int32>()() = num_shards_;
    TF_RETURN_IF_ERROR(NodeDefBuilder(name_prefix+"/num_shards", "Const")
                       .Device(device_name)
                       .Attr("dtype", DT_INT32)
                       .Attr("value", num_shards_val)
                       .Finalize(&num_shards_def));
    num_shards_node = g->AddNode(num_shards_def, &s);

    return s;
  }

  Status CreateMergeAndUnPackCacheCKPTResourceOp(std::unique_ptr<Graph>& g,
           const std::string& name_prefix, const std::string& device_name,
           const std::vector<Node*>& switch_nodes, Node* load_ckpt_node,
           Node*& unpack_node) {
    // Create merge op.
    std::vector<NodeDefBuilder::NodeOut> src_list;
    for (Node* n : switch_nodes) {
      src_list.emplace_back(n->name(), 1, n->output_type(1));
    }
    src_list.emplace_back(load_ckpt_node->name(), 0,
                          load_ckpt_node->output_type(0));
    NodeDef merge_def;
    TF_RETURN_IF_ERROR(NodeDefBuilder(name_prefix + "/Merge", "Merge")
                       .Device(device_name)
                       .Input(src_list)
                       .Finalize(&merge_def));
    Status s;
    Node* merge_node = g->AddNode(merge_def, &s);
    TF_RETURN_IF_ERROR(s);
    // Connect input edges for 'Merge' op.
    for (size_t i = 0; i < switch_nodes.size(); i++) {
      Node* n = switch_nodes[i];
      g->AddEdge(n, 1, merge_node, i);
    }
    g->AddEdge(load_ckpt_node, 0, merge_node, switch_nodes.size());

    // Create UnPackCacheCKPTResource node.
    NodeDef unpack_node_def;
    TF_RETURN_IF_ERROR(NodeDefBuilder(name_prefix + "/UnPackCacheCKPTResource",
                                      "UnPackCacheCKPTResource")
                       .Device(device_name)
                       .Input(merge_node->name(), 0, merge_node->output_type(0))
                       .Finalize(&unpack_node_def));
    unpack_node = g->AddNode(unpack_node_def, &s);

    return s;
  }

  // Variables.
  struct RelativeNodes relative_nodes_;
  std::string cache_path_;
  int64 cache_ckpt_replica_;
  int num_shards_;
  std::string local_storage_type_;
};
} // Endof namespace cache_ckpt_pass

class CacheCKPTPass : public GraphOptimizationPass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override {
    if (!IsEnableCacheCKPT()) {
      return Status::OK();
    }

    VLOG(0) << "CacheCKPTPass: Enable Cache CKPT.";
    cache_path_ = GetLocalCacheCKPTPath();
    cache_ckpt_replica_ = GetCacheCKPTReplica();
    local_storage_type_ = GetLocalCacheCKPTStorageTypeFromEnvVar();
    remote_storage_type_ = GetRemoteCacheCKPTStorageTypeFromEnvVar();

    Graph* graph = options.graph->get();
    if (graph == nullptr) {
      return errors::Internal("a graph should be available.");
    }
    std::unique_ptr<Graph> new_graph(new Graph(OpRegistry::Global()));
    CopyGraph(*graph, new_graph.get());

    struct cache_ckpt_pass::RelativeNodes relative_nodes;
    bool find_relative_nodes = GetRelativeNodes(new_graph, relative_nodes);
    if (!find_relative_nodes) {
      VLOG(0) << "CacheCKPTPass: Failed to get relative nodes, skip this pass.";
      return Status::OK();
    }

    // Get num_shards.
    int num_shards = 0;
    GetNumShards(relative_nodes, num_shards);
    if (cache_ckpt_replica_ >= num_shards) {
      VLOG(0) << "CacheCKPTPass: environment variable 'CACHE_CKPT_REPLICA' (val: "
              << cache_ckpt_replica_
              << ") should be < num_shards (val: " << num_shards
              << "), skip this pass.";
      return Status::OK();
    }

    // Add SaveCacheCKPT subgraph.
    save_helper.reset(
      new cache_ckpt_pass::AddSaveCacheCKPTSubGraphHelper(relative_nodes,
                              cache_path_, cache_ckpt_replica_, num_shards,
                              local_storage_type_, remote_storage_type_));
    TF_RETURN_IF_ERROR(save_helper->Run(new_graph));

    // Add RestoreCacheCKPT subgraph.
    restore_helper.reset(
      new cache_ckpt_pass::AddRestoreCacheCKPTSubGraphHelper(relative_nodes,
                             cache_path_, cache_ckpt_replica_, num_shards,
                             local_storage_type_));
    TF_RETURN_IF_ERROR(restore_helper->Run(new_graph));

    options.graph->swap(new_graph);

    VLOG(0) << "CacheCKPTPass: Enable Cache CKPT Success.";
    return Status::OK();
 }

 private:
  // Functions.
  bool GetRelativeNodes(std::unique_ptr<Graph>& graph,
                        struct cache_ckpt_pass::RelativeNodes& relative_nodes) {
    relative_nodes.merge_checkpoints_node = nullptr;
    bool find_merge_ckpt_node = false;
    for (Node* n : graph->op_nodes()) {
      if (n->type_string() == "ShardedFilename") {
        relative_nodes.sharded_filename_nodes.insert(n);
      } else if (n->type_string() == "RestoreV2" ||
                 n->type_string() == "KvResourceImportV2" ||
                 n->type_string() == "KvResourceImportV3") {
        relative_nodes.restore_nodes.insert(n);
      } else if (n->type_string() == "MergeV2Checkpoints") {
        if (find_merge_ckpt_node) {
          VLOG(1) << "CacheCKPTPass: The graph contains multiple MergeV2Checkpoints nodes.";
          return false;
        }
        relative_nodes.merge_checkpoints_node = n;
        find_merge_ckpt_node = true;
      }
    }

    auto& sharded_nodes = relative_nodes.sharded_filename_nodes;
    auto& restore_nodes = relative_nodes.restore_nodes;
    if (sharded_nodes.size() < 2) {
      VLOG(1) << "CacheCKPTPass: Failed to find ShardedFilename op, the num is "
              << sharded_nodes.size() << ", which should be >= 2.";
      return false;
    } else if (restore_nodes.size() < 2) {
      VLOG(1) << "CacheCKPTPass: Failed to find RestoreV2 or KvResourceImportV2, the num is "
              << restore_nodes.size() << ", which should be >= 2.";
      return false;
    } else if (!find_merge_ckpt_node) {
      VLOG(1) << "CacheCKPTPass: Failed to find MergeV2Checkpoints.";
      return false;
    }

    auto& merge_node = relative_nodes.merge_checkpoints_node;
    Status s = merge_node->input_node(1, &relative_nodes.ckpt_path_prefix_node);
    if (!s.ok()) {
      VLOG(1) << "CacheCKPTPass: Failed to find cache path prefix node.";
      return false;
    }

    return true;
  }

  Status GetNumShards(
            const struct cache_ckpt_pass::RelativeNodes& relative_nodes,
            int& num_shards) {
    Node* sharded_filename_node = \
      *(relative_nodes.sharded_filename_nodes.begin());
    Node* num_shards_node = nullptr;
    Tensor num_shards_tensor;
    TF_RETURN_IF_ERROR(sharded_filename_node->input_node(2, &num_shards_node));
    TF_RETURN_IF_ERROR(GetNodeAttr(num_shards_node->attrs(), "value",
                                   &num_shards_tensor));
    num_shards = num_shards_tensor.scalar<int32>()();
    CHECK_EQ(relative_nodes.sharded_filename_nodes.size(), num_shards);

    return Status::OK();
  }

  bool IsEnableCacheCKPT() {
    bool is_enable_cache_ckpt = false;
    TF_CHECK_OK(ReadBoolFromEnvVar("ENABLE_CACHE_CKPT", true,
                                   &is_enable_cache_ckpt));
    return is_enable_cache_ckpt;
  }

  string GetLocalCacheCKPTPath() {
    std::string cache_path;
    TF_CHECK_OK(ReadStringFromEnvVar("CACHE_CKPT_PATH",
                  "/tmp/tf_fault_tolerance/cache_ckpt", &cache_path));
    return cache_path;
  }

  int64 GetCacheCKPTReplica() {
    int64 cache_ckpt_replica = 0;
    TF_CHECK_OK(ReadInt64FromEnvVar("CACHE_CKPT_REPLICA", 1,
                                    &cache_ckpt_replica));
    return cache_ckpt_replica;
  }

  // Variables.
  std::string cache_path_;
  int64 cache_ckpt_replica_;
  std::string local_storage_type_;
  std::string remote_storage_type_;
  std::unique_ptr<cache_ckpt_pass::AddSaveCacheCKPTSubGraphHelper> save_helper;
  std::unique_ptr<cache_ckpt_pass::AddRestoreCacheCKPTSubGraphHelper>
    restore_helper;
};

REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_PLACEMENT, 0,
                      CacheCKPTPass);

} // End of namespace tensorflow
