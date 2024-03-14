#!/usr/bin/env python

# Copyright 2023 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import time

import types
from collections import defaultdict

from dynamic_embedding_server.proto import elastic_training_pb2, elastic_training_pb2_grpc
from dynamic_embedding_server.python import redistribution_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import device as tf_device
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops import array_ops, control_flow_ops, resource_variable_ops
from tensorflow.python.training import session_run_hook, training_util
from tensorflow.python.training.basic_session_run_hooks import SecondOrStepTimer
from tensorflow.python import pywrap_tensorflow as tf_session

import grpc

ELASTIC_SUBGRAPH_INIT = "elastic_subgraph_init"
ELASTIC_SUBGRAPH_IMPORT = "elastic_subgraph_import"
ELASTIC_IMPORT_SCOPE = "elastic_import"
AIMASTER_ADDR = "AIMASTER_ADDR"

def create(self):
  # pylint: disable=protected-access
  self._session = None
  opts = tf_session.TF_NewSessionOptions(target=self._target, config=self._config)
  try:
    self._session = tf_session.TF_NewSessionRef(self._graph._c_graph, opts)
    self._closed = False
  finally:
    tf_session.TF_DeleteSessionOptions(opts)
  # pylint: enable=protected-access


class ElasticTrainingHook(session_run_hook.SessionRunHook):
  """Connect AIMaster periodly to check if needed to do scaling
    ```python
      dynamic_embedding_server.ElasticTrainingHook()
    ```

  """
  def __init__(self, check_scale_secs=None, check_scale_steps=1000):
    self._task_type = json.loads(os.environ.get('TF_CONFIG', ""))["task"]["type"]
    self._aimaster_addr = os.environ.get(AIMASTER_ADDR, "")
    self._base_device = tf_device.DeviceSpec.from_string("/job:ps/task:0/device:CPU:0")
    self._timer = SecondOrStepTimer(
        every_secs=check_scale_secs, every_steps=check_scale_steps)

  def after_create_session(self, session, coord):
    session.create = types.MethodType(create, session)
    global_step = session.run(self._global_step_tensor)
    self._timer.update_last_triggered_step(global_step)

  def begin(self):
    self._global_step_tensor = training_util._get_or_create_global_step_read()  # pylint: disable=protected-access
    if self._global_step_tensor is None:
      raise RuntimeError(
          "Global step should be created to use ElasticTrainingHook.")
    self.init_op = [control_flow_ops.no_op(ELASTIC_SUBGRAPH_INIT)]
    self.partition_num_ph = array_ops.placeholder(dtypes.int32, name='partition_num')
    self.op_list = self._init_ev_repartition_op()
    self.op_list.extend(self._init_var_repartition_op())
    with ops.control_dependencies(self.op_list):
      self.import_op = [control_flow_ops.no_op(ELASTIC_SUBGRAPH_IMPORT)]
    #reserve for synchronization between chief and worker
    self._add_sync_graph()
    try:
      graph = ops.get_default_graph()
      main_init = graph.get_operation_by_name("init")
      main_init._add_control_input(variables.variables_initializer(ops.get_collection(ops.GraphKeys.LOCAL_VARIABLES)))
    except:
      logging.info("skipping add init ops")
  
  def before_run(self, run_context):  # pylint: disable=unused-argument
    return session_run_hook.SessionRunArgs(self._global_step_tensor)

  def after_run(self, run_context, run_values):
    stale_global_step = run_values.results
    if self._timer.should_trigger_for_step(stale_global_step):
      # get the real value after train op.
      global_step = run_context.session.run(self._global_step_tensor)
      if self._timer.should_trigger_for_step(global_step):
        self._timer.update_last_triggered_step(global_step)
        self._check_scale(run_context)

  def _check_scale(self, run_context):
    channel = grpc.insecure_channel(self._aimaster_addr)
    req = elastic_training_pb2.IsReadyScalingRequest()
    task_id = 0 if self._task_type == "chief" else 1
    req.task_index = task_id
    _stub = elastic_training_pb2_grpc.ElasticTrainingServiceStub(channel)
    try:
      resp = _stub.IsReadyScaling(req)
      #reset sync variable
      run_context.session.run(self.sync_reset)
      if resp.code == elastic_training_pb2.OK:
        if resp.scaling_action == elastic_training_pb2.SCALING_UP:
          logging.info("[DES]: start scaling up process.")
          partition_num = resp.ps_num
          run_context.session.close()
          _req = elastic_training_pb2.ReadyToUpdateRequest()
          _stub.ReadyToUpdate(_req)
          run_context.session.create()
          if self._task_type == "chief":
            run_context.session.run(self.init_op)
            run_context.session.run(self.import_op,
                                    feed_dict={self.partition_num_ph: partition_num})
            run_context.session.run(self.sync_ok)
          else:
            while False == run_context.session.run(self.read_sync):
              logging.info("chief not ready, wait for 5 secs")
              time.sleep(5)
          logging.info("[DES]: finish scaling up process.")
        elif resp.scaling_action == elastic_training_pb2.SCALING_DOWN:
          logging.info("[DES]: start scaling down process.")
          partition_num = resp.ps_num
          if self._task_type == "chief":
            run_context.session.run(self.import_op,
                                    feed_dict={self.partition_num_ph: partition_num})
            run_context.session.run(self.sync_ok)
          else:
            while False == run_context.session.run(self.read_sync):
              logging.info("chief not ready, wait for 5 secs")
              time.sleep(5)
            #reset sync variable
            run_context.session.run(self.sync_reset)
          run_context.session.close()
          _req = elastic_training_pb2.ReadyToUpdateRequest()
          _stub.ReadyToUpdate(_req)
          run_context.session.create()
          self._rewrite_op_device()
          
          if self._task_type == "chief":
            try:
              run_context.session.run(self.init_op)
              run_context.session.run(self.sync_ok)
            except Exception as error:
              logging.info("failed to run init_ops in graph...")
          else:
            while False == run_context.session.run(self.read_sync):
              logging.info("chief not ready, wait for 5 secs")
              time.sleep(5)
          logging.info("[DES]: finish scaling down process.")
        else:
          logging.info("No need to do scaling.")
    except Exception as error:
      logging.warning("AIMASTER UNAVAILABLE. {}".format(error))

  def _init_var_repartition_op(self):
    op_list = []
    ev_list = ops.get_collection(ops.GraphKeys.EMBEDDING_VARIABLES)
    variable_list = [x for x in ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES) if x not in ev_list]
    var_meta_map = defaultdict(list)
    def process_var_name(embedding_var):
      ev_name = embedding_var.name.split(":")[0]
      device_name = embedding_var.device
      idx = device_name.find("task:")
      post_name = device_name[idx:]
      post_idx = post_name.find("/")
      if post_idx == -1:
        device_id = int(device_name[idx+len("task:"):])
      else:
        device_id = int(post_name[len("task:"):post_idx])
      idx = ev_name.find("part_")
      if idx != -1:
        post_idx = ev_name[idx:].find("/")
        part_len = len("part_")
        if (post_idx == -1):
          pre_name = ev_name[:idx-1]
          var_idx = int(ev_name[idx+part_len:])
        else:
          pre_name = ev_name[:idx-1] + ev_name[idx:][post_idx:]
          var_idx = int(ev_name[idx:][part_len:post_idx])
        return True, pre_name, device_id, var_idx
      else:
        new_op = None
        if "global_step" not in embedding_var.name:
          new_op = self._add_tmp_variable(embedding_var)
        return False, new_op, 0, 0

    with ops.name_scope(ELASTIC_IMPORT_SCOPE):
      for var in variable_list:
        flag, pre_name, device_id, var_idx = process_var_name(var)
        if flag:
          var_meta_map[pre_name].append((device_id, var_idx, var))
        elif pre_name is not None:
          op_list.append(pre_name)

      for var_list in var_meta_map.values():
        var_list.sort(key= lambda x: x[0])
        var_read = [var[2] for var in var_list]
        if len(var_list) == 1:
          embedding_var = var_list[0][2]
          new_op = self._add_tmp_variable(embedding_var)
          op_list.append(new_op)
        else:
          read_value = array_ops.concat(var_read, axis=0) #partition_axis
          for idx, var_meta in enumerate(var_list):
            if resource_variable_ops.is_resource_variable(var_list[idx][2]):
              op_list.append(redistribution_ops.re_assign_resource( \
                  var_list[idx][2], read_value, self.partition_num_ph, idx, len(var_list)))
            else:
              op_list.append(redistribution_ops.re_assign( \
                  var_list[idx][2]._ref(), read_value, self.partition_num_ph, idx, len(var_list)))
    return op_list

  def _init_ev_repartition_op(self):
    op_list = []
    tot_ev_list = ops.get_collection(ops.GraphKeys.EMBEDDING_VARIABLES)
    var_map = {}
    primary_to_opt_map = defaultdict(list)
    import_storage_map = defaultdict(lambda: defaultdict(list))
    def process_ev_name(embedding_var):
      ev_name = embedding_var.name.split(":")[0]
      idx = ev_name.find("part_")
      post_idx = ev_name[idx:].find("/")
      primary_name = None
      if post_idx == -1:
        ev_name = ev_name[:idx-1]
      else:
        ev_name = ev_name[:idx-1] + ev_name[idx:][post_idx:]
        primary_name = ev_name[:idx-1]
      return ev_name, primary_name

    for embedding_var in tot_ev_list:
      # pylint: disable=protected-access
      is_partitioned_ev = not isinstance(embedding_var._save_slice_info, str)
      save_slice_info = embedding_var._save_slice_info
      partition_num = save_slice_info.full_shape[0] if is_partitioned_ev else 1
      partition_id = save_slice_info.var_offset[0] if is_partitioned_ev else 0
      # pylint: enable=protected-access
      pre_name, primary_name = process_ev_name(embedding_var)
      if pre_name not in var_map:
        var_map[pre_name] = [None for _ in range(partition_num)]
        if primary_name is not None:
          primary_to_opt_map[primary_name].append(pre_name)

      var_map[pre_name][partition_id] = embedding_var

    for primary_name, opt_name_list in primary_to_opt_map.items():
      primary_ev_list = list(filter(lambda x: x != None, var_map[primary_name]))
      partition_num = len(primary_ev_list)
      tmp_op_list = [[] for _ in range(partition_num)]
      for opt_name in opt_name_list:
        opt_ev_list = list(filter(lambda x: x != None, var_map[opt_name]))
        import_storage_map[opt_name]["keys"] = [None for _ in range(partition_num)]
        import_storage_map[opt_name]["values"]  = [None for _ in range(partition_num)]
        import_storage_map[opt_name]["versions"] = [None for _ in range(partition_num)]
        import_storage_map[opt_name]["freqs"] = [None for _ in range(partition_num)]
        for idx, embedding_var in enumerate(opt_ev_list):
          key_type = dtypes.as_dtype(embedding_var.handle.op.get_attr("Tkeys"))
          dtype = dtypes.as_dtype(embedding_var.handle.op.get_attr("dtype"))
          unneeded_ids, unneeded_values, unneeded_versions, unneeded_freqs = \
              redistribution_ops.kv_resource_filter(embedding_var,
                                                    self.partition_num_ph,
                                                    key_type,
                                                    dtype,
                                                    partition_id=idx)
          tmp_op_list[idx].append(unneeded_ids)
          import_storage_map[opt_name]["keys"][idx] = unneeded_ids
          import_storage_map[opt_name]["values"][idx] = unneeded_values
          import_storage_map[opt_name]["versions"][idx] = unneeded_versions
          import_storage_map[opt_name]["freqs"][idx] = unneeded_freqs

      import_storage_map[primary_name]["keys"] = [None for _ in range(partition_num)]
      import_storage_map[primary_name]["values"]  = [None for _ in range(partition_num)]
      import_storage_map[primary_name]["versions"] = [None for _ in range(partition_num)]
      import_storage_map[primary_name]["freqs"] = [None for _ in range(partition_num)]
      for idx, embedding_var in enumerate(primary_ev_list):
        key_type = dtypes.as_dtype(embedding_var.handle.op.get_attr("Tkeys"))
        dtype = dtypes.as_dtype(embedding_var.handle.op.get_attr("dtype"))
        with ops.control_dependencies(tmp_op_list[idx]):
          unneeded_ids, unneeded_values, unneeded_versions, unneeded_freqs = \
              redistribution_ops.kv_resource_filter(embedding_var, 
                                                    self.partition_num_ph,
                                                    key_type, dtype,
                                                    partition_id=idx)
        import_storage_map[primary_name]["keys"][idx] = unneeded_ids
        import_storage_map[primary_name]["values"][idx] = unneeded_values
        import_storage_map[primary_name]["versions"][idx] = unneeded_versions
        import_storage_map[primary_name]["freqs"][idx] = unneeded_freqs

    for embedding_var in tot_ev_list:
      pre_name, _ = process_ev_name(embedding_var)
      # pylint: disable=protected-access
      is_partitioned_ev = not isinstance(embedding_var._save_slice_info, str)
      partition_id = 0
      save_slice_info = embedding_var._save_slice_info
      if save_slice_info is not None:
        partition_id = save_slice_info.var_offset[0] if is_partitioned_ev else 0
      # pylint: enable=protected-access
      # 1st input of ImportStorageOp is itself, it will be skipped in Kernel
      tmp_key_list = import_storage_map[pre_name]["keys"].copy()
      tmp_value_list = import_storage_map[pre_name]["values"].copy()
      tmp_version_list = import_storage_map[pre_name]["versions"].copy()
      tmp_freq_list = import_storage_map[pre_name]["freqs"].copy()

      imported_keys = [tmp_key_list.pop(partition_id)]
      imported_values = [tmp_value_list.pop(partition_id)]
      imported_versions = [tmp_version_list.pop(partition_id)]
      imported_freqs = [tmp_freq_list.pop(partition_id)]

      for i in range(len(tmp_key_list)):
        imported_keys.append(tmp_key_list[i])
        imported_values.append(tmp_value_list[i])
        imported_versions.append(tmp_version_list[i])
        imported_freqs.append(tmp_freq_list[i])

      op_list.append(redistribution_ops.kv_resource_mul_import( \
          embedding_var, imported_keys, imported_values,
          imported_versions, imported_freqs, partition_id=partition_id))

    return op_list
  
  def _add_sync_graph(self):
    with ops.device(self._base_device):
      self.sync_variable = variables.VariableV1(False,
                                                trainable=False,
                                                dtype=dtypes.bool,
                                                name="worker_sync",
                                                collections=[ops.GraphKeys.LOCAL_VARIABLES])
      self.read_sync = self.sync_variable.read_value()
      self.sync_ok = self.sync_variable.assign(True)
      self.sync_reset = self.sync_variable.assign(False)

  def _add_tmp_variable(self, embedding_var):
    with ops.device(self._base_device):
      if embedding_var.dtype == dtypes.int64_ref:
        init_value = [1]
        p = variables.VariableV1(init_value,
                                  dtype=dtypes.int64,
                                  name=embedding_var.name[:-2],
                                  collections=[ops.GraphKeys.LOCAL_VARIABLES])
      else:
        init_value = [1.0]
        p = variables.VariableV1(init_value, 
                                  dtype=dtypes.float32,
                                  name=embedding_var.name[:-2],
                                  collections=[ops.GraphKeys.LOCAL_VARIABLES])
    new_op = state_ops.assign(p, 
                              embedding_var.read_value(),
                              validate_shape=False)
    return new_op

  def _rewrite_op_device(self):
    graph = ops.get_default_graph()
    op_list = graph.get_operations()
    for op in op_list: # pylint: disable=invalid-name
      op_device = op.device
      device_spec = tf_device.DeviceSpec.from_string(op_device)
      if device_spec.task is not None and device_spec.task >= partition_num:
        device_spec.task = 0
        op._set_device(device_spec.to_string())
