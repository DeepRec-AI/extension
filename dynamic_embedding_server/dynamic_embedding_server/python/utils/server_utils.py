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
"""Utils for elastic training server."""

import time
import json
import os
import grpc

from deeprec_master.proto import elastic_training_pb2, \
    elastic_training_pb2_grpc, error_code_pb2
from tensorflow.python.estimator import run_config as run_config_lib
from tensorflow.python.framework import errors_impl
from dynamic_embedding_server.python.utils.logger import logger
from dynamic_embedding_server.python.utils.retry import retry
from tensorflow.python.training import checkpoint_utils

@retry(retries=5)
def estimate_model_resource(checkpoint_dir):
  """estimate model size in checkpoint"""
  try:
    reader = checkpoint_utils.load_checkpoint(checkpoint_dir)
  except ValueError as e:
    logger.warning('load_checkpoint error: {}.'.format(e))
    return 0, False
  except (errors_impl.NotFoundError, errors_impl.InvalidArgumentError):
    logger.warning('load_checkpoint error, Not found.')
    return 0, False

  variable_map = reader.get_variable_to_shape_map()
  dtype_map = reader.get_variable_to_dtype_map()

  names = sorted(variable_map.keys())
  total_size = 0
  has_ev = False
  for name in names:
    if name.endswith("-keys"):
      has_ev = True
    var_elements = 1
    for i in variable_map[name]:
      var_elements *= i
    total_size += var_elements * dtype_map[name].size
  return total_size, has_ev

@retry(retries=5)
def report_estimated_ps_resource(stub, has_ev, total_size):
  req = elastic_training_pb2.PsResourceRequest()
  req.ps_num = -1
  req.ps_memory = total_size
  req.has_ev = has_ev
  resp = stub.EstimatePSResource(req)
  if resp.code != error_code_pb2.OK:
    logger.error("report_estimated_ps_resource error.")

@retry(retries=5)
def set_tfconfig(stub, tf_config):
  req = elastic_training_pb2.MasterRequest()
  req.msg = tf_config
  resp = stub.SetTFConfig(req)
  if resp.code != error_code_pb2.OK:
    logger.error("set_tfconfig error.")

@retry(retries=5)
def get_tfconfig(stub):
  req = elastic_training_pb2.MasterRequest()
  resp = stub.GetTFConfig(req)
  if resp.code == error_code_pb2.OK:
    return resp.msg
  else:
    logger.error("get_tfconfig error.")
    return "notfound"

def estimator_prepare_or_wait_for_cluster(config):
  """worker-0: set TF_CONFIG environment to aimaster
     other role: get TF_CONFIG environment from aimaster"""
  aimaster_addr = os.getenv('DEEPRECMASTER_ADDR')
  if aimaster_addr is None:
    logger.fatal("aimaster addr not set")
  channel = grpc.insecure_channel(aimaster_addr)
  stub = elastic_training_pb2_grpc.ElasticTrainingServiceStub(channel)

  new_tfconfig = os.environ.get("TF_CONFIG", "{}")

  if (config.task_type == run_config_lib.TaskType.CHIEF
      or (config.task_type == run_config_lib.TaskType.WORKER
          and config.task_id == 0)):
    logger.info("Estimate ps resource on model %s ...", config.model_dir)
    total_size, has_ev = estimate_model_resource(config.model_dir)
    report_estimated_ps_resource(stub, has_ev, total_size)
  elif config.task_type == run_config_lib.TaskType.PS and config.task_id == 0:
    new_tfconfig = os.environ.get("TF_CONFIG", "{}")
    logger.info("Set TF_CONFIG: %s, talking to %s,",
                 new_tfconfig, aimaster_addr)
    set_tfconfig(stub, new_tfconfig)
  else:
    pass

  new_tfconfig = get_tfconfig(stub)
  logger.info("NEW_TF_CONFIG is %s", new_tfconfig)
  while new_tfconfig.find("notfound") != -1:
    logger.info("Get TF_CONFIG fail: %s, when talking to %s, retrying...",
                 new_tfconfig, aimaster_addr)
    time.sleep(5)
    new_tfconfig = get_tfconfig(stub)
  tfconfig = os.environ.get("TF_CONFIG", "{}")

  logger.info("Get old TF_CONFIG: %s ", tfconfig)
  logger.info('Get TF_CONFIG success: %s', new_tfconfig)

  os.environ["TF_CONFIG"] = new_tfconfig
  new_tfconfig = json.loads(new_tfconfig)
  old_tfconfig = json.loads(tfconfig)
  cluster_spec = new_tfconfig.get("cluster", {})
  cluster_spec[config.task_type][config.task_id] = old_tfconfig["cluster"][config.task_type][config.task_id]
  return cluster_spec


def prepare_or_wait_for_cluster(task_type, task_id, model_dir):
  """worker-0: set TF_CONFIG environment to aimaster
     other role: get TF_CONFIG environment from aimaster"""
  aimaster_addr = os.getenv('DEEPRECMASTER_ADDR')
  if aimaster_addr is None:
    logger.fatal("aimaster addr not set")
  channel = grpc.insecure_channel(aimaster_addr)
  stub = elastic_training_pb2_grpc.ElasticTrainingServiceStub(channel)

  
  new_tfconfig = os.environ.get("TF_CONFIG", "{}")

  if (task_type == run_config_lib.TaskType.CHIEF
      or (task_type == run_config_lib.TaskType.WORKER
          and task_id == 0)):
    logger.info("Estimate ps resource on model %s ...", model_dir)
    total_size, has_ev = estimate_model_resource(model_dir)
    report_estimated_ps_resource(stub, has_ev, total_size)
  elif task_type == run_config_lib.TaskType.PS and task_id == 0:
    logger.info("Set TF_CONFIG: %s, talking to %s,",
                 new_tfconfig, aimaster_addr)
    set_tfconfig(stub, new_tfconfig)
  else:
    pass

  new_tfconfig = get_tfconfig(stub)
  logger.info("NEW_TF_CONFIG is %s", new_tfconfig)
  while new_tfconfig.find("notfound") != -1:
    logger.info("Get TF_CONFIG fail: %s, when talking to %s, retrying...",
                 new_tfconfig, aimaster_addr)
    time.sleep(5)
    new_tfconfig = get_tfconfig(stub)
  tfconfig = os.environ.get("TF_CONFIG", "{}")

  logger.info("Get old TF_CONFIG: %s ", tfconfig)
  logger.info('Get TF_CONFIG success: %s', new_tfconfig)

  os.environ["TF_CONFIG"] = new_tfconfig
  new_tfconfig = json.loads(new_tfconfig)
  old_tfconfig = json.loads(tfconfig)
  cluster_spec = new_tfconfig.get("cluster", {})
  cluster_spec[task_type][task_id] = old_tfconfig["cluster"][task_type][task_id]
  #TODO(FIXME:JUNQI) in fuxi worker is scheduler slower ps when
  if task_type == "worker":
    if len(cluster_spec["ps"]) != len(old_tfconfig["cluster"]["ps"]):
      cluster_spec = old_tfconfig.get("cluster", {})
  return cluster_spec
