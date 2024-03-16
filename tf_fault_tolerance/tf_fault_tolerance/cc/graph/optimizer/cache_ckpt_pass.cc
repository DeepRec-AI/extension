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
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/util/env_var.h"

#include "tf_fault_tolerance/cc/ops/cache_ckpt_ops.h"
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
      remote_storage_type_(remote_storage_type), fetch_op_target_(nullptr),
      cancel_op_target_(nullptr), resume_op_target_(nullptr) {
    ckpt_path_prefix_node_ = relative_nodes.ckpt_path_prefix_node;
    TF_CHECK_OK(GetAsyncCKPTPathPutTimeOutMillisecond());
  }

  ~AddSaveCacheCKPTSubGraphHelper() {}

  Status Run(std::unique_ptr<Graph>& g) {
    // Get the map of 'shard_id' to 'sharded_filename_node'.
    auto& sharded_filename_nodes = relative_nodes_.sharded_filename_nodes;
    std::vector<Node*> shard_to_sharded_filename_node;
    TF_RETURN_IF_ERROR(
      GetShardToShardedFilenameNodeMap(sharded_filename_nodes,
                                       shard_to_sharded_filename_node));

    const std::string async_name_prefix = "save/AsyncSaveCacheCKPT";
    enable_async_saving_ = \
      TryToFindAsyncSavingRelativeNodes(g, async_name_prefix);
    Node* ckpt_path_put_node = nullptr;
    if (enable_async_saving_) {
      // Create async saving node.
      TF_RETURN_IF_ERROR(CreateAsyncSavingNodes(g, async_name_prefix,
                                                ckpt_path_put_node));
    } else {
      LOG(WARNING) << "CacheCKPTPass: enable async saving cache ckpt failed, "
                   << "use sync saving instead.";
    }

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

    if (enable_async_saving_) {
      g->AddControlEdge(merge_checkpoints_node, ckpt_path_put_node);
      g->AddControlEdge(ckpt_path_put_node, save_run_node);

      for (size_t i = 0; i < shard_to_generate_node.size(); i++) {
        Node* generate_node = shard_to_generate_node[i];
        for (Node* n : shard_to_backup_node[i]) {
          g->AddControlEdge(generate_node, n);
          g->AddControlEdge(n, fetch_op_target_);
        }
      }
    } else {
      for (size_t i = 0; i < shard_to_generate_node.size(); i++) {
        Node* generate_node = shard_to_generate_node[i];
        g->AddControlEdge(merge_checkpoints_node, generate_node);
        for (Node* n : shard_to_backup_node[i]) {
          g->AddControlEdge(generate_node, n);
          g->AddControlEdge(n, save_run_node);
        }
      }
    }

    return Status::OK();
  }

 private:
  // Functions.
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

  bool TryToFindAsyncSavingRelativeNodes(std::unique_ptr<Graph>& g,
                                         const std::string& name_prefix) {
    bool find_fetch = false;
    bool find_cancel = false;
    bool find_resume = false;

    for (Node* n : g->op_nodes()) {
      if (n->type_string() != "NoOp") {
        continue;
      }

      if (n->name() == name_prefix+"/async_fetch_op") {
        fetch_op_target_ = n;
        find_fetch = true;
      } else if (n->name() == name_prefix+"/async_cancel_op") {
        cancel_op_target_ = n;
        find_cancel = true;
      } else if (n->name() == name_prefix+"/async_resume_op") {
        resume_op_target_ = n;
        find_resume = true;
      }
    }

    return find_fetch && find_cancel && find_resume;
  }

  Status CreateAsyncSavingNodes(std::unique_ptr<Graph>& g,
                                const std::string& name_prefix,
                                Node*& ckpt_path_put_node) {
    std::string device_name = ckpt_path_prefix_node_->assigned_device_name();
    // Create ckpt path put node.
    std::string put_node_name = name_prefix + "/ItemBufferPut";
    TF_RETURN_IF_ERROR(NodeBuilder(put_node_name, "ItemBufferPut")
                       .Input(ckpt_path_prefix_node_, 0)
                       .Attr("shared_name", "async_cache_ckpt")
                       .Attr("is_overwritable", true)
                       .Attr("timeout_millis", timeout_millis_)
                       .Finalize(g.get(), &ckpt_path_put_node));
    ckpt_path_put_node->set_assigned_device_name(device_name);

    // Create ckpt path take node.
    std::string take_node_name = name_prefix + "/ItemBufferTake";
    Node* ckpt_path_take_node = nullptr;
    TF_RETURN_IF_ERROR(NodeBuilder(take_node_name, "ItemBufferTake")
                       .Attr("dtype", ckpt_path_prefix_node_->output_type(0))
                       .Attr("shared_name", "async_cache_ckpt")
                       .Attr("is_overwritable", true)
                       .Finalize(g.get(), &ckpt_path_take_node));
    ckpt_path_take_node->set_assigned_device_name(device_name);
    ckpt_path_prefix_node_ = ckpt_path_take_node;

    // Create resume node.
    std::string resume_node_name = name_prefix + "/ItemBufferResume";
    Node* ckpt_path_resume_node = nullptr;
    TF_RETURN_IF_ERROR(NodeBuilder(resume_node_name, "ItemBufferSetState")
                       .Attr("is_cancelled", false)
                       .Attr("shared_name", "async_cache_ckpt")
                       .Attr("is_overwritable", true)
                       .Finalize(g.get(), &ckpt_path_resume_node));
    ckpt_path_resume_node->set_assigned_device_name(device_name);
    g->AddControlEdge(ckpt_path_resume_node, resume_op_target_);

    // Create cancel node.
    std::string cancel_node_name = name_prefix + "/ItemBufferCancel";
    Node* ckpt_path_cancel_node = nullptr;
    TF_RETURN_IF_ERROR(NodeBuilder(cancel_node_name, "ItemBufferSetState")
                       .Attr("is_cancelled", true)
                       .Attr("shared_name", "async_cache_ckpt")
                       .Attr("is_overwritable", true)
                       .Finalize(g.get(), &ckpt_path_cancel_node));
    ckpt_path_cancel_node->set_assigned_device_name(device_name);
    g->AddControlEdge(ckpt_path_cancel_node, cancel_op_target_);

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
    Node* generate_node = nullptr;
    TF_RETURN_IF_ERROR(NodeBuilder(node_name, "GenerateCacheCKPT")
                       .Input(ckpt_path_prefix_node_, 0)
                       .Input(cache_path_node, 0)
                       .Input(shard_node, 0)
                       .Input(num_shards_node, 0)
                       .Attr("ckpt_storage_type", local_storage_type_)
                       .Finalize(g.get(), &generate_node));
    generate_node->set_assigned_device_name(device_name);
    shard_to_generate_node[shard] = generate_node;

    return Status::OK();
  }

  Status CreateInputNodesForGenerateCacheCKPTNode(std::unique_ptr<Graph>& g,
           const std::string& name_prefix, const std::string& device_name,
           const int shard, Node*& cache_path_node, Node*& shard_node,
           Node*& num_shards_node) {
    // cache_path node.
    Tensor cache_path_val(DT_STRING, TensorShape({}));
    cache_path_val.scalar<string>()() = \
      cache_path_ + "_" + std::to_string(shard);
    TF_RETURN_IF_ERROR(NodeBuilder(name_prefix+"/cache_path", "Const")
                       .Attr("dtype", DT_STRING)
                       .Attr("value", cache_path_val)
                       .Finalize(g.get(), &cache_path_node));
    cache_path_node->set_assigned_device_name(device_name);

    // shard node.
    Tensor shard_val(DT_INT32, TensorShape({}));
    shard_val.scalar<int32>()() = shard;
    TF_RETURN_IF_ERROR(NodeBuilder(name_prefix+"/shard", "Const")
                       .Attr("dtype", DT_INT32)
                       .Attr("value", shard_val)
                       .Finalize(g.get(), &shard_node));
    shard_node->set_assigned_device_name(device_name);

    // num_shards node.
    Tensor num_shards_val(DT_INT32, TensorShape({}));
    num_shards_val.scalar<int32>()() = num_shards_;
    TF_RETURN_IF_ERROR(NodeBuilder(name_prefix+"/num_shards", "Const")
                       .Attr("dtype", DT_INT32)
                       .Attr("value", num_shards_val)
                       .Finalize(g.get(), &num_shards_node));
    num_shards_node->set_assigned_device_name(device_name);

    return Status::OK();
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
      Node* cache_path_node = nullptr;
      Tensor cache_path_val(DT_STRING, TensorShape({}));
      cache_path_val.scalar<string>()() = \
        cache_path_ + "_" + std::to_string(id);
      TF_RETURN_IF_ERROR(NodeBuilder(node_name+"/cache_path", "Const")
                         .Attr("dtype", DT_STRING)
                         .Attr("value", cache_path_val)
                         .Finalize(g.get(), &cache_path_node));
      cache_path_node->set_assigned_device_name(device_name);

      Node* backup_node = nullptr;
      TF_RETURN_IF_ERROR(NodeBuilder(node_name, "BackupRemoteCacheCKPT")
                         .Input(cache_path_node, 0)
                         .Input(generate_node, 0)
                         .Input(generate_node, 1)
                         .Input(generate_node, 2)
                         .Attr("ckpt_storage_type", remote_storage_type_)
                         .Finalize(g.get(), &backup_node));
      backup_node->set_assigned_device_name(device_name);
      shard_to_backup_node[id].insert(backup_node);
    }

    return Status::OK();
  }

  Status GetAsyncCKPTPathPutTimeOutMillisecond() {
    // default value is 5min.
    return ReadInt64FromEnvVar("ASYNC_CKPT_PATH_PUT_TIMEOUT_MS", 300000,
                               &timeout_millis_);
  }

  // Variables.
  struct RelativeNodes relative_nodes_;
  std::string cache_path_;
  int64 cache_ckpt_replica_;
  int num_shards_;
  std::string local_storage_type_;
  std::string remote_storage_type_;
  Node* ckpt_path_prefix_node_;
  Node* fetch_op_target_;
  Node* cancel_op_target_;
  Node* resume_op_target_;
  int64 timeout_millis_;
  bool enable_async_saving_;
};

//------------------------------------------------------------------------------
// Helper class to add restore cache ckpt subgraph.
class AddRestoreCacheCKPTSubGraphHelper {
 public:
  AddRestoreCacheCKPTSubGraphHelper(const struct RelativeNodes& relative_nodes,
                                    const std::string& cache_path,
                                    const int64 cache_ckpt_replica,
                                    const int num_shards,
                                    const std::string& local_storage_type,
                                    const std::string& remote_storage_type)
    : relative_nodes_(relative_nodes), cache_path_(cache_path),
      cache_ckpt_replica_(cache_ckpt_replica), num_shards_(num_shards),
      local_storage_type_(local_storage_type),
      remote_storage_type_(remote_storage_type) {}

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
    Node* switch_node = nullptr;
    TF_RETURN_IF_ERROR(NodeBuilder(node_name + "/Swtich", "Switch")
                       .Input(check_local_node, 1)
                       .Input(check_local_node, 0)
                       .Finalize(g.get(), &switch_node));
    switch_node->set_assigned_device_name(device_name);

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
    TF_RETURN_IF_ERROR(NodeBuilder(node_name, "CheckLocalCacheCKPT")
                       .Input(ckpt_prefix_node, 0)
                       .Input(shard_node, 0)
                       .Input(num_shards_node, 0)
                       .Attr("shared_name", "cache_ckpt")
                       .Finalize(g.get(), &check_local_node));
    check_local_node->set_assigned_device_name(device_name);

    return Status::OK();
  }

  Status CreateInputNodesForCheckLocalCacheCKPTOp(
           std::unique_ptr<Graph>& g, const std::string& name_prefix,
           const std::string device_name, const int shard, Node*& shard_node,
           Node*& num_shards_node) {
    Tensor shard_val(DT_INT32, TensorShape({}));
    shard_val.scalar<int32>()() = shard;
    TF_RETURN_IF_ERROR(NodeBuilder(name_prefix+"/shard", "Const")
                       .Attr("dtype", DT_INT32)
                       .Attr("value", shard_val)
                       .Finalize(g.get(), &shard_node));
    shard_node->set_assigned_device_name(device_name);

    Tensor num_shards_val(DT_INT32, TensorShape({}));
    num_shards_val.scalar<int32>()() = num_shards_;
    TF_RETURN_IF_ERROR(NodeBuilder(name_prefix+"/num_shards", "Const")
                       .Attr("dtype", DT_INT32)
                       .Attr("value", num_shards_val)
                       .Finalize(g.get(), &num_shards_node));
    num_shards_node->set_assigned_device_name(device_name);

    return Status::OK();
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
    Node* switch_node = nullptr;
    TF_RETURN_IF_ERROR(NodeBuilder(repatriate_name + "/Switch", "Switch")
                       .Input(ckpt_prefix_node, 0)
                       .Input(in_check_node, 0)
                       .Finalize(g.get(), &switch_node));
    switch_node->set_assigned_device_name(device_name);

    // Create RepatriateRemoteCacheCKPT op.
    TF_RETURN_IF_ERROR(CreateRepatriateRemoteCacheCKPTOp(g, repatriate_name,
                         device_name, shard, switch_node, repatriate_node));

    return Status::OK();
  }

  Status CreateRepatriateRemoteCacheCKPTOp(std::unique_ptr<Graph>& g,
           const std::string& send_node_name, const std::string& device_name,
           const int shard, Node* switch_node, Node*& send_node) {
    // Create input nodes of RepatriateRemoteCacheCKPT.
    Node* shard_node = nullptr;
    Tensor shard_val(DT_INT32, TensorShape({}));
    shard_val.scalar<int32>()() = shard;
    TF_RETURN_IF_ERROR(NodeBuilder(send_node_name+"/shard", "Const")
                       .Attr("dtype", DT_INT32)
                       .Attr("value", shard_val)
                       .Finalize(g.get(), &shard_node));
    shard_node->set_assigned_device_name(device_name);

    Node* num_shards_node = nullptr;
    Tensor num_shards_val(DT_INT32, TensorShape({}));
    num_shards_val.scalar<int32>()() = num_shards_;
    TF_RETURN_IF_ERROR(NodeBuilder(send_node_name+"/num_shards", "Const")
                       .Attr("dtype", DT_INT32)
                       .Attr("value", num_shards_val)
                       .Finalize(g.get(), &num_shards_node));
    num_shards_node->set_assigned_device_name(device_name);

    // Create RepatriateRemoteCacheCKPT op.
    bool output_is_path = RemoteStorageIsFileSystem();
    TF_RETURN_IF_ERROR(NodeBuilder(send_node_name, "RepatriateRemoteCacheCKPT")
                       .Input(switch_node, 0)
                       .Input(shard_node, 0)
                       .Input(num_shards_node, 0)
                       .Attr("output_is_path", output_is_path)
                       .Finalize(g.get(), &send_node));
    send_node->set_assigned_device_name(device_name);

    return Status::OK();
  }

  bool RemoteStorageIsFileSystem() {
    return remote_storage_type_ == StorageType::kPosixFileType;
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
    Node* switch_node = nullptr;
    TF_RETURN_IF_ERROR(NodeBuilder(get_ckpt_name + "/Switch", "Switch")
                       .Input(get_remote_node, 1)
                       .Input(get_remote_node, 0)
                       .Finalize(g.get(), &switch_node));
    switch_node->set_assigned_device_name(device_name);

    switch_nodes.push_back(switch_node);

    return Status::OK();
  }

  Status CreateGetRemoteCacheCKPTOp(std::unique_ptr<Graph>& g,
           const std::string& node_name, const std::string& device_name,
           const int shard, Node* repatriate_node, Node*& get_remote_node) {
    // Create inputs for GetRemoteCacheCKPT op.
    Node* cache_path_node = nullptr;
    Tensor cache_path_val(DT_STRING, TensorShape({}));
    cache_path_val.scalar<string>()() = \
      cache_path_ + "_" + std::to_string(shard);
    TF_RETURN_IF_ERROR(NodeBuilder(node_name+"/cache_path", "Const")
                       .Attr("dtype", DT_STRING)
                       .Attr("value", cache_path_val)
                       .Finalize(g.get(), &cache_path_node));
    cache_path_node->set_assigned_device_name(device_name);

    // Create GetRemoteCacheCKPT op.
    Node* ckpt_prefix_node = relative_nodes_.ckpt_path_prefix_node;
    TF_RETURN_IF_ERROR(NodeBuilder(node_name, "GetRemoteCacheCKPT")
                       .Input(ckpt_prefix_node, 0)
                       .Input(cache_path_node, 0)
                       .Input(repatriate_node, 0)
                       .Input(repatriate_node, 1)
                       .Input(repatriate_node, 2)
                       .Input(repatriate_node, 3)
                       .Attr("shared_name", "cache_ckpt")
                       .Attr("ckpt_storage_type", local_storage_type_)
                       .Finalize(g.get(), &get_remote_node));
    get_remote_node->set_assigned_device_name(device_name);

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
    Node* switch_node = nullptr;
    TF_RETURN_IF_ERROR(NodeBuilder(load_node_name+"/Switch", "Switch")
                       .Input(ckpt_path_prefix_node, 0)
                       .Input(check_ckpt_node, 0)
                       .Finalize(g.get(), &switch_node));
    switch_node->set_assigned_device_name(device_name);

    // Create LoadCKPTFromPathFile op.
    TF_RETURN_IF_ERROR(NodeBuilder(load_node_name, "LoadCKPTFromFilePath")
                       .Input(switch_node, 0)
                       .Input(shard_node, 0)
                       .Input(num_shards_node, 0)
                       .Finalize(g.get(), &load_ckpt_node));
    load_ckpt_node->set_assigned_device_name(device_name);

    return Status::OK();
  }

  Status CreateInputNodesForLoadCKPTFromFilePathOp(
           std::unique_ptr<Graph>& g, const std::string& name_prefix,
           const std::string device_name, const int shard, Node*& shard_node,
           Node*& num_shards_node) {
    Tensor shard_val(DT_INT32, TensorShape({}));
    shard_val.scalar<int32>()() = shard;
    TF_RETURN_IF_ERROR(NodeBuilder(name_prefix+"/shard", "Const")
                       .Attr("dtype", DT_INT32)
                       .Attr("value", shard_val)
                       .Finalize(g.get(), &shard_node));
    shard_node->set_assigned_device_name(device_name);

    Tensor num_shards_val(DT_INT32, TensorShape({}));
    num_shards_val.scalar<int32>()() = num_shards_;
    TF_RETURN_IF_ERROR(NodeBuilder(name_prefix+"/num_shards", "Const")
                       .Attr("dtype", DT_INT32)
                       .Attr("value", num_shards_val)
                       .Finalize(g.get(), &num_shards_node));
    num_shards_node->set_assigned_device_name(device_name);

    return Status::OK();
  }

  Status CreateMergeAndUnPackCacheCKPTResourceOp(std::unique_ptr<Graph>& g,
           const std::string& name_prefix, const std::string& device_name,
           const std::vector<Node*>& switch_nodes, Node* load_ckpt_node,
           Node*& unpack_node) {
    // Create merge op.
    std::vector<NodeBuilder::NodeOut> src_list;
    for (Node* n : switch_nodes) {
      src_list.emplace_back(n, 1);
    }
    src_list.emplace_back(load_ckpt_node, 0);

    Node* merge_node = nullptr;
    TF_RETURN_IF_ERROR(NodeBuilder(name_prefix + "/Merge", "Merge")
                       .Input(src_list)
                       .Finalize(g.get(), &merge_node));
     merge_node->set_assigned_device_name(device_name);

    // Create UnPackCacheCKPTResource node.
    TF_RETURN_IF_ERROR(NodeBuilder(name_prefix + "/UnPackCacheCKPTResource",
                                   "UnPackCacheCKPTResource")
                       .Input(merge_node, 0)
                       .Finalize(g.get(), &unpack_node));
    unpack_node->set_assigned_device_name(device_name);

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
} // Endof namespace cache_ckpt_pass

//------------------------------------------------------------------------------
// Add cache ckpt save and restore subgraph.
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
                             local_storage_type_, remote_storage_type_));
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

//------------------------------------------------------------------------------
// Replace the specific 'Send'/'Recv' op with
// 'SliceSend'/'SliceRecv' or 'FileSliceSend'/'FileSliceRecv' op.
class ReplaceTransferOpWithSliceTransferOpPass : public GraphOptimizationPass {
 public:
  ReplaceTransferOpWithSliceTransferOpPass() : GraphOptimizationPass() {
    TF_CHECK_OK(GetSliceSizeFromEnvVar());
    TF_CHECK_OK(GetSliceTransferTimeOutMillisecond());
    TF_CHECK_OK(GetFileSliceRecvTMPDir());

    VLOG(2) << "ReplaceTransferOp Pass config: [slice_size: (" << slice_size_
            << ") bytes, recv_timeout: (" << timeout_ms_
            << ") ms, file_recv_tmp_dir: ("
            << file_slice_recv_tmp_dir_ << ")].";
  }

  Status Run(const GraphOptimizationPassOptions& options) override {
    if (!IsEnableReplaceTransferOp()) {
      return Status::OK();
    }

    if (options.partition_graphs == nullptr) {
      return errors::Internal("partition graphs should be available.");
    }

    for (auto& pg : *(options.partition_graphs)) {
      Graph* graph = (pg.second).get();
      std::unique_ptr<Graph> new_graph(new Graph(OpRegistry::Global()));
      CopyGraph(*graph, new_graph.get());
      TF_RETURN_IF_ERROR(ModifyGraph(new_graph));
      (pg.second).swap(new_graph);
    }

    VLOG(2) << "Replace transfer op with slice transfer op success.";

    return Status::OK();
  }

 private:
  // Functions.
  bool IsEnableReplaceTransferOp() {
    bool need_replace_send = false;
    TF_CHECK_OK(ReadBoolFromEnvVar("ENABLE_CACHE_CKPT_REPLACE_TRANSFER_OP",
                                   true, &need_replace_send));
    return need_replace_send;
  }

  Status GetSliceSizeFromEnvVar() {
    // default value is 64 MB.
    return ReadInt64FromEnvVar("SLICE_TRANS_SIZE", 0x4000000, &slice_size_);
  }

  Status GetSliceTransferTimeOutMillisecond() {
    // default value is 'never timout'.
    return ReadInt64FromEnvVar("SLICE_TRANS_TIMEOUT_MS", 0, &timeout_ms_);
  }

  Status ModifyGraph(std::unique_ptr<Graph>& graph) {
    for (Node* n : graph->op_nodes()) {
      if (n->IsSend() && n->type_string() != "_RefSend") {
        TF_RETURN_IF_ERROR(TryToReplaceSendOp(graph, n));
      } else if (n->IsRecv() && n->type_string() != "_RefRecv") {
        TF_RETURN_IF_ERROR(TryToReplaceRecvOp(graph, n));
      }
    }
    return Status::OK();
  }

  Status TryToReplaceSendOp(std::unique_ptr<Graph>& g, Node* n) {
    // Get input edge.
    CHECK_EQ(n->num_inputs(), 1);
    const Edge* e = nullptr;
    TF_RETURN_IF_ERROR(n->input_edge(0, &e));
    Node* src_node = e->src();
    int out_idx = e->src_output();

    if (!NeedReplaceSendOp(src_node, out_idx)) {
      return Status::OK();
    }

    std::string send_op = "_FileSliceSend";
    if (src_node->type_string() == "RepatriateRemoteCacheCKPT") {
      const AttrValue* output_is_path_attr = \
        src_node->attrs().Find("output_is_path");
      CHECK_NE(output_is_path_attr, nullptr);
      bool output_is_path = output_is_path_attr->b();
      if (!output_is_path) {
        send_op = "_SliceSend";
      }
    }

    auto attrs = n->attrs();
    Node* new_send_node = nullptr;
    TF_RETURN_IF_ERROR(NodeBuilder(n->name(), send_op)
                       .Input(src_node, out_idx)
                       .Attr("tensor_name", *(attrs.Find("tensor_name")))
                       .Attr("send_device", *(attrs.Find("send_device")))
                       .Attr("send_device_incarnation",
                             *(attrs.Find("send_device_incarnation")))
                       .Attr("recv_device", *(attrs.Find("recv_device")))
                       .Attr("client_terminated",
                             *(attrs.Find("client_terminated")))
                       .Attr("slice_size", slice_size_)
                       .Finalize(g.get(), &new_send_node));
    new_send_node->set_assigned_device_name(n->assigned_device_name());

    TF_RETURN_IF_ERROR(g->UpdateEdge(src_node, out_idx, new_send_node, 0));
    g->RemoveNode(n);

    return Status::OK();
  }

  bool NeedReplaceSendOp(const Node* src_node, const int out_idx) {
    using namespace CacheCKPTOp;

    const std::string& src_op = src_node->type_string();

    if (src_op == "GenerateCacheCKPT" && \
        (out_idx == GenerateCacheCKPTOp::MetaCKPTOutputIdx || \
         out_idx == GenerateCacheCKPTOp::DataCKPTOutputIdx)) {
      return true;
    } else if (src_op == "RepatriateRemoteCacheCKPT" && \
               (out_idx == RepatriateRemoteCacheCKPTOp::MetaCKPTOutputIdx || \
                out_idx == RepatriateRemoteCacheCKPTOp::DataCKPTOutputIdx)) {
      return true;
    }

    return false;
  }

  Status TryToReplaceRecvOp(std::unique_ptr<Graph>& g, Node* n) {
    if (!NeedReplaceRecvOp(n)) {
      return Status::OK();
    }

    auto attrs = n->attrs();
    Node* new_recv_node = nullptr;
    TF_RETURN_IF_ERROR(NodeBuilder(n->name(), "_FileSliceRecv")
                       .Attr("tensor_name", *(attrs.Find("tensor_name")))
                       .Attr("send_device", *(attrs.Find("send_device")))
                       .Attr("send_device_incarnation",
                             *(attrs.Find("send_device_incarnation")))
                       .Attr("recv_device", *(attrs.Find("recv_device")))
                       .Attr("client_terminated",
                             *(attrs.Find("client_terminated")))
                       .Attr("recv_dir", file_slice_recv_tmp_dir_)
                       .Attr("slice_size", slice_size_)
                       .Attr("timeout_ms", timeout_ms_)
                       .Finalize(g.get(), &new_recv_node));
    new_recv_node->set_assigned_device_name(n->assigned_device_name());

    for (auto e : n->out_edges()) {
      Node* dst_node = e->dst();
      int in_idx = e->dst_input();
      TF_RETURN_IF_ERROR(g->UpdateEdge(new_recv_node, 0, dst_node, in_idx));
    }

    g->RemoveNode(n);
    return Status::OK();
  }

  bool NeedReplaceRecvOp(const Node* n) {
    using namespace CacheCKPTOp;

    for (auto e : n->out_edges()) {
      const std::string& dst_op = e->dst()->type_string();
      int in_idx = e->dst_input();
      if (dst_op == "BackupRemoteCacheCKPT" && \
          (in_idx == BackupRemoteCacheCKPTOp::MetaCKPTInputIdx || \
           in_idx == BackupRemoteCacheCKPTOp::DataCKPTInputIdx)) {
        return true;
      } else if (dst_op == "GetRemoteCacheCKPT" && \
                 (in_idx == GetRemoteCacheCKPTOp::MetaCKPTInputIdx || \
                  in_idx == GetRemoteCacheCKPTOp::DataCKPTInputIdx)) {
        return true;
      }
    }

    return false;
  }

  Status GetFileSliceRecvTMPDir() {
    return ReadStringFromEnvVar("CACHE_CKPT_FILE_SLICE_RECV_TMP_DIR",
             "/tmp/tf_fault_tolerance/tmp_file_slice_recv_dir",
             &file_slice_recv_tmp_dir_);
  }

  // Variable.
  int64 slice_size_;
  int64 timeout_ms_;
  string file_slice_recv_tmp_dir_;
};

REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_PARTITIONING, 0,
                      ReplaceTransferOpWithSliceTransferOpPass);

} // End of namespace tensorflow
