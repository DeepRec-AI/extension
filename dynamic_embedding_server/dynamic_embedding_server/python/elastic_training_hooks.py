#!/usr/bin/env python

# Copyright 2024 The DeepRec Authors. All Rights Reserved.
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

from deeprec_master.proto import elastic_training_pb2, \
    elastic_training_pb2_grpc
from dynamic_embedding_server.python import redistribution_ops

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import device as tf_device
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from dynamic_embedding_server.python.utils.logger import logger
from tensorflow.python.ops import array_ops, control_flow_ops, resource_variable_ops
from tensorflow.python.training import session_run_hook, training_util
from tensorflow.python.training.basic_session_run_hooks import SecondOrStepTimer
from tensorflow.python import pywrap_tensorflow as tf_session

import grpc

ELASTIC_SUBGRAPH_INIT = "elastic_subgraph_init"
ELASTIC_SUBGRAPH_IMPORT = "elastic_subgraph_import"
ELASTIC_IMPORT_SCOPE = "elastic_import"
ELASTIC_SUBGRAPH_DESTROY = "elastic_subgraph_destroy"
DEEPRECMASTER_ADDR = "DEEPRECMASTER_ADDR"


def create(self):
  # pylint: disable=protected-access
  self._session = None
  opts = tf_session.TF_NewSessionOptions(
      target=self._target, config=self._config)
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
  def __init__(self, check_scale_secs=None, check_scale_steps=None):
    self._task_type = json.loads(os.environ.get('TF_CONFIG', ""))[
        "task"]["type"]
    self._aimaster_addr = os.environ.get(DEEPRECMASTER_ADDR, "")
    self._base_cpu_device = tf_device.DeviceSpec.from_string(
        "/job:ps/task:0/device:CPU:0")
    self._base_gpu_device = tf_device.DeviceSpec.from_string(
        "/job:ps/task:0/device:GPU:0")
    if check_scale_secs is None and check_scale_steps is None:
      check_scale_steps = 1000
    self._timer = SecondOrStepTimer(
        every_secs=check_scale_secs, every_steps=check_scale_steps)
    self._channel = grpc.insecure_channel(self._aimaster_addr)
    self._initialized = False

  def after_create_session(self, session, coord):
    session.create = types.MethodType(create, session)
    global_step = session.run(self._global_step_tensor)
    self._timer.update_last_triggered_step(global_step)

  def begin(self):
    if self._initialized:
      return
    self._global_step_tensor = training_util._get_or_create_global_step_read(
    )  # pylint: disable=protected-access
    if self._global_step_tensor is None:
      raise RuntimeError(
          "Global step should be created to use ElasticTrainingHook.")
    self.init_op = [control_flow_ops.no_op(ELASTIC_SUBGRAPH_INIT)]
    self.del_op_list = []
    with ops.device(self._base_cpu_device):
      self.partition_num_ph = array_ops.placeholder(
          dtypes.int32, shape=[], name='partition_num')
    self.op_list = self._init_ev_repartition_op()
    self.op_list.extend(self._init_var_repartition_op())
    with ops.control_dependencies(self.op_list):
      self.import_op = [control_flow_ops.no_op(ELASTIC_SUBGRAPH_IMPORT)]
    with ops.control_dependencies(self.del_op_list):
      self.destroy_op = [control_flow_ops.no_op(
          ELASTIC_SUBGRAPH_DESTROY)]
    # reserve for synchronization between chief and worker
    self._add_sync_graph()
    try:
      graph = ops.get_default_graph()
      # ops.GraphKeys.INIT_OP
      main_init = graph.get_operation_by_name("init")
      main_init._add_control_input(variables.variables_initializer(
          ops.get_collection(ops.GraphKeys.LOCAL_VARIABLES)))
    except:
      logger.info("skipping add init ops")
    self._initialized = True

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
    req = elastic_training_pb2.IsReadyScalingRequest()
    task_id = 0 if self._task_type == "chief" else 1
    req.task_index = task_id
    _stub = elastic_training_pb2_grpc.ElasticTrainingServiceStub(
        self._channel)
    try:
      resp = _stub.IsReadyScaling(req)
      # reset sync variable
      run_context.session.run(self.sync_reset)
      if resp.code == elastic_training_pb2.OK:
        if resp.scaling_action == elastic_training_pb2.SCALING_UP:
          logger.info("[DES]: start scaling up process.")
          partition_num = resp.ps_num
          run_context.session.close()
          _req = elastic_training_pb2.ReadyToUpdateRequest()
          _stub.ReadyToUpdate(_req)
          run_context.session.create()
          logger.info("[DES]: start repartition.")
          if self._task_type == "chief":
            run_context.session.run(self.init_op,
                                    feed_dict={self.partition_num_ph: partition_num})
            run_context.session.run(self.import_op,
                                    feed_dict={self.partition_num_ph: partition_num})
            run_context.session.run(self.destroy_op)
            run_context.session.run(self.sync_ok)
          else:
            while False == run_context.session.run(self.read_sync):
              logger.info("chief not ready, wait for 5 secs")
              time.sleep(5)
          logger.info("[DES]: finish scaling up process.")
        elif resp.scaling_action == elastic_training_pb2.SCALING_DOWN:
          logger.info("[DES]: start scaling down process.")
          partition_num = resp.ps_num
          if self._task_type == "chief":
            run_context.session.run(self.import_op,
                                    feed_dict={self.partition_num_ph: partition_num})
            run_context.session.run(self.sync_ok)
          else:
            while False == run_context.session.run(self.read_sync):
              logger.info("chief not ready, wait for 5 secs")
              time.sleep(5)
            # reset sync variable
            run_context.session.run(self.sync_reset)
          run_context.session.close()
          _req = elastic_training_pb2.ReadyToUpdateRequest()
          _stub.ReadyToUpdate(_req)
          run_context.session.create()
          self._rewrite_op_device(partition_num)

          if self._task_type == "chief":
            try:
              run_context.session.run(self.init_op)
              run_context.session.run(self.destroy_op)
              run_context.session.run(self.sync_ok)
            except Exception as error:
              logger.info("failed to run init_ops in graph...")
          else:
            while False == run_context.session.run(self.read_sync):
              logger.info("chief not ready, wait for 5 secs")
              time.sleep(5)
          logger.info("[DES]: finish scaling down process.")
        else:
          logger.info("No need to do scaling.")
    except Exception as error:
      logger.warning("AIMASTER UNAVAILABLE. {}".format(error))

  def _init_var_repartition_op(self):
    op_list = []
    ev_list = self._get_embedding_variable_list()
    variable_list = [x for x in ops.get_collection(
        ops.GraphKeys.GLOBAL_VARIABLES) if x not in ev_list]
    variable_list.extend(ops.get_collection(ops.GraphKeys.LOCAL_VARIABLES))
    # list represent for: [device_idx, part_idx, var]
    var_meta_map = defaultdict(list)

    def process_var_name(embedding_var):
      ev_name = embedding_var.name.split(":")[0]
      device_name = embedding_var.device
      idx = device_name.find("task:")
      post_name = device_name[idx:]
      post_idx = post_name.find("/")
      no_device = False
      if post_idx == -1:
        try:
          device_id = int(device_name[idx+len("task:"):])
        except ValueError:
          no_device = True
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
        if no_device:
          device_id = var_idx
        return True, pre_name, device_id, var_idx
      else:
        new_op, del_op = self._add_tmp_variable(embedding_var)
        return False, new_op, del_op, 0

    with ops.name_scope(ELASTIC_IMPORT_SCOPE):
      for var in variable_list:
        flag, pre_name, device_id, var_idx = process_var_name(var)
        if flag:
          var_meta_map[pre_name].append((device_id, var_idx, var))
        elif pre_name is not None:
          op_list.append(pre_name)
          self.del_op_list.append(device_id)

      for var_list in var_meta_map.values():
        var_list.sort(key=lambda x: x[1])
        var_read = [var[2] for var in var_list]
        if len(var_list) == 0:
          continue

        if len(var_list) == 1:
          embedding_var = var_list[0][2]
          new_op, del_op = self._add_tmp_variable(embedding_var)
          if new_op is not None:
            op_list.append(new_op)
            self.del_op_list.append(del_op)
        else:
          try:
            read_value = array_ops.concat(
              var_read, axis=0)  # partition_axis
          except ValueError:
            read_value = array_ops.stack(var_read)
          for var_meta in var_list:
            if resource_variable_ops.is_resource_variable(var_meta[2]):
              op_list.append(redistribution_ops.re_assign_resource(
                  var_meta[2], read_value, self.partition_num_ph,
                  var_meta[1], var_meta[0], len(var_list)))
            else:
              op_list.append(redistribution_ops.re_assign(
                  var_meta[2]._ref(
                  ), read_value, self.partition_num_ph,
                  var_meta[1], var_meta[0], len(var_list)))
    return op_list

  def _init_ev_repartition_op(self):
    op_list = []
    var_map = {}
    primary_to_opt_map = defaultdict(list)
    self._prepare_ev_map(primary_to_opt_map, var_map)
    import_storage_map = defaultdict(lambda: defaultdict(list))
    self._make_filter_op(primary_to_opt_map, var_map, import_storage_map)
    self._make_import_op(primary_to_opt_map, var_map,
                         import_storage_map, op_list)

    return op_list

  def _add_sync_graph(self):
    with ops.device(self._base_cpu_device):
      self.sync_variable = variables.VariableV1(False,
                                                trainable=False,
                                                dtype=dtypes.bool,
                                                name="worker_sync",
                                                collections=[ops.GraphKeys.LOCAL_VARIABLES])
      self.read_sync = self.sync_variable.read_value()
      self.sync_ok = self.sync_variable.assign(True)
      self.sync_reset = self.sync_variable.assign(False)

  def _get_embedding_variable_list(self):
    try:
      ev_collection = ops.get_collection(
          ops.GraphKeys.EMBEDDING_VARIABLES)
      if len(ev_collection) == 0:
        return []
      tot_ev_dict = json.loads(ev_collection[0])
      tot_ev_list = []
      tot_var = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      for var in tot_var:
        idx = var.name.rfind("/part_")
        pre_name = var.name[:idx]
        if pre_name in tot_ev_dict:
          tot_ev_list.append(var)
    except:
      tot_ev_list = ops.get_collection(ops.GraphKeys.EMBEDDING_VARIABLES)
    return tot_ev_list

  def _prepare_ev_map(self, primary_to_opt_map, var_map):
    tot_ev_list = self._get_embedding_variable_list()

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

    # Init primary to opt variable meta
    for embedding_var in tot_ev_list:
      # pylint: disable=protected-access
      is_partitioned_ev = not isinstance(
          embedding_var._save_slice_info, str)
      partition_num = embedding_var._save_slice_info.full_shape[0] if is_partitioned_ev else 1
      partition_id = embedding_var._save_slice_info.var_offset[0] if is_partitioned_ev else 0
      # pylint: enable=protected-access
      pre_name, primary_name = process_ev_name(embedding_var)
      if pre_name not in var_map:
        var_map[pre_name] = [None for _ in range(partition_num)]
        if primary_name is not None:
          primary_to_opt_map[primary_name].append(pre_name)

      var_map[pre_name][partition_id] = embedding_var

  def _make_import_op(self, primary_to_opt_map, var_map, import_storage_map, op_list):
    for primary_name, opt_name_list in primary_to_opt_map.items():
      partition_num = len(
          list(filter(lambda x: x != None, var_map[primary_name])))
      tmp_op_list = [None for _ in range(partition_num)]
      self._import_impl(import_storage_map, primary_name,
                        var_map, op_list, tmp_op_list)
      for opt_name in opt_name_list:
        self._import_impl(import_storage_map, opt_name,
                          var_map, op_list, tmp_op_list)

  def _import_impl(self, import_storage_map, ev_name, var_map, op_list, tmp_op_list):
    ev_list = list(filter(lambda x: x != None, var_map[ev_name]))
    partition_num = len(ev_list)
    if partition_num == 1:
      return
    for idx, embedding_var in enumerate(ev_list):
      partition_id = idx if partition_num > 1 else 0
      device_id = self._get_actual_device(embedding_var)
      if device_id == -1:
        device_id = idx
      # pylint: enable=protected-access
      # 1st input of ImportStorageOp is itself, it will be skipped in Kernel
      tmp_key_list = import_storage_map[ev_name]["keys"][partition_id][:]
      tmp_value_list = import_storage_map[ev_name]["values"][partition_id][:]
      tmp_version_list = import_storage_map[ev_name]["versions"][partition_id][:]
      tmp_freq_list = import_storage_map[ev_name]["freqs"][partition_id][:]

      imported_keys = [tmp_key_list.pop(idx)]
      imported_values = [tmp_value_list.pop(idx)]
      imported_versions = [tmp_version_list.pop(idx)]
      imported_freqs = None
      imported_freqs = [tmp_freq_list.pop(idx)]

      for i in range(len(tmp_key_list)):
        imported_keys.append(tmp_key_list[i])
        imported_values.append(tmp_value_list[i])
        imported_versions.append(tmp_version_list[i])
        imported_freqs.append(tmp_freq_list[i])

      # with ops.device(embedding_var.device):
      if tmp_op_list[partition_id] is None:
        a = redistribution_ops.kv_resource_mul_import(
            embedding_var, self.partition_num_ph, imported_keys, imported_values,
            imported_versions, imported_freqs, partition_id=partition_id, device_id=device_id, partition_nums=partition_num)
        tmp_op_list[partition_id] = a
      else:
        with ops.control_dependencies([tmp_op_list[partition_id]]):
          a = redistribution_ops.kv_resource_mul_import(
              embedding_var, self.partition_num_ph, imported_keys, imported_values,
              imported_versions, imported_freqs, partition_id=partition_id, device_id=device_id, partition_nums=partition_num)
      op_list.append(a)

  def _make_filter_op(self, primary_to_opt_map, var_map, import_storage_map):
    for primary_name, opt_name_list in primary_to_opt_map.items():
      partition_num = len(
          list(filter(lambda x: x != None, var_map[primary_name])))
      tmp_op_list = [[] for _ in range(partition_num)]
      for opt_name in opt_name_list:
        self._filter_impl(import_storage_map,
                          opt_name, var_map, tmp_op_list)
      self._filter_impl(import_storage_map, primary_name,
                        var_map, tmp_op_list, True)

  def _filter_impl(self, import_storage_map, ev_name, var_map, tmp_op_list, is_primary=False):
    ev_list = list(filter(lambda x: x != None, var_map[ev_name]))
    partition_num = len(ev_list)
    if partition_num == 1:
      return
    import_storage_map[ev_name]["keys"] = [[]
                                           for _ in range(partition_num)]
    import_storage_map[ev_name]["values"] = [[]
                                             for _ in range(partition_num)]
    import_storage_map[ev_name]["versions"] = [[]
                                               for _ in range(partition_num)]
    import_storage_map[ev_name]["freqs"] = [[]
                                              for _ in range(partition_num)]
    for idx, embedding_var in enumerate(ev_list):
      key_type = dtypes.as_dtype(
          embedding_var.handle.op.get_attr("Tkeys"))
      dtype = dtypes.as_dtype(embedding_var.handle.op.get_attr("dtype"))
      device_id = self._get_actual_device(embedding_var)
      if device_id == -1:
        device_id = idx
      if is_primary:
        with ops.control_dependencies(tmp_op_list[idx]):
          unneeded_structs = \
              redistribution_ops.kv_resource_filter(embedding_var,
                                                    self.partition_num_ph,
                                                    key_type, dtype,
                                                    partition_id=idx,
                                                    partition_num=partition_num,
                                                    device_id=device_id)
      else:
        unneeded_structs = \
            redistribution_ops.kv_resource_filter(embedding_var,
                                                  self.partition_num_ph,
                                                  key_type, dtype,
                                                  partition_id=idx,
                                                  partition_num=partition_num,
                                                  device_id=device_id)
        tmp_op_list[idx].extend(unneeded_structs.keys)
      for part_id in range(partition_num):
        import_storage_map[ev_name]["keys"][part_id].append(
            unneeded_structs.keys[part_id])
        import_storage_map[ev_name]["values"][part_id].append(
            unneeded_structs.values[part_id])
        import_storage_map[ev_name]["versions"][part_id].append(
            unneeded_structs.versions[part_id])
        import_storage_map[ev_name]["freqs"][part_id].append(
            unneeded_structs.freqs[part_id])

  def _add_tmp_variable(self, embedding_var):
    op_device = embedding_var.device
    device_spec = tf_device.DeviceSpec.from_string(op_device)
    if device_spec.device_type is not None and device_spec.device_type in ("GPU", "gpu"):
      device_context = self._base_gpu_device
    else:
      device_context = self._base_cpu_device
    with ops.device(device_context):
      # use_resource = resource_variable_ops.is_resource_variable(embedding_var)
      if embedding_var.dtype == dtypes.int64_ref or embedding_var.dtype == dtypes.int64:
        init_var = array_ops.ones(embedding_var.shape, dtypes.int64)
        dtype = dtypes.int64
      elif embedding_var.dtype == dtypes.float32_ref or embedding_var.dtype == dtypes.float32:
        init_var = array_ops.ones(embedding_var.shape, dtypes.float32)
        dtype = dtypes.float32
      else:
        logger.warning("============= datatype not in int64 or float32 {}".format(embedding_var.name))
        return None, None

      p = variables.VariableV1(init_var,
                               dtype=dtype,
                               name=embedding_var.name[:-2],
                               use_resource=True,
                               collections=[ops.GraphKeys.LOCAL_VARIABLES])
      new_op = state_ops.assign(p,
                                embedding_var.read_value(),
                                validate_shape=False)
      del_op = resource_variable_ops.destroy_resource_op(p.handle)
    return new_op, del_op

  def _get_actual_device(self, variable):
    device_name = variable.device
    device_spec = tf_device.DeviceSpec.from_string(device_name)
    if device_spec.task is not None:
      return device_spec.task
    else:
      return -1

  def _rewrite_op_device(self, partition_num):
    graph = ops.get_default_graph()
    op_list = graph.get_operations()
    for op in op_list:  # pylint: disable=invalid-name
      op_device = op.device
      device_spec = tf_device.DeviceSpec.from_string(op_device)
      if device_spec.task is not None and device_spec.task >= partition_num:
        device_spec.task = 0
        op._set_device(device_spec.to_string())
