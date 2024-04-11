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
"""Cluster utilities for elastic training."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

from dynamic_embedding_server.python.utils.logger import logger


class DistributeConfig(object):  # pylint: disable=useless-object-inheritance
  """AIMaster distributed job configs."""

  def __init__(self):
    self._cluster_spec = None
    self._get_tfjob_cluster_config()
    logger.info("task_name:{}, task_index:{}, is_master:{}".format(
        self._task_name, self._task_index, self._is_master))

  @property
  def task_name(self):
    return self._task_name

  @property
  def task_index(self):
    return self._task_index

  @property
  def is_master_task(self):
    return self._is_master

  @property
  def cluster_spec(self):
    return self._cluster_spec

  def _get_tfjob_cluster_config(self):
    """Get tf job cluster config."""
    if self._try_get_from_launch_tf_parameter():
      # TF_CONFIG is regenerated from the launch parameters on fuxi.
      # Referene: https://yuque.antfin-inc.com/pai-user/manual/zya6r3
      return

    tf_config_str = os.getenv('TF_CONFIG')
    if tf_config_str is None:
      raise RuntimeError("Not found `TF_CONFIG` env variable.")

    logger.info("TF_CONFIG {}".format(tf_config_str))
    try:
      tf_config = json.loads(tf_config_str)
    except Exception as ex:
      raise ValueError("Invalid json format TF_CONFIG:{},"  # pylint: disable=raise-missing-from
                       "exception:{}".format(tf_config_str, ex))

    assert 'cluster' in tf_config and 'task' in tf_config, \
        "Invalid TF_CONFIG: {}, no cluster or task key.".format(tf_config)

    task = tf_config['task']
    assert 'type' in task and 'index' in task, \
        "Invalid TF_CONFIG task: {}, no type or index key.".format(
            tf_config)
    task_type = task['type']
    task_index = int(task['index'])

    cluster = tf_config['cluster']
    chief_hosts = cluster.get('chief', None)
    worker_hosts = cluster.get('worker', None)
    if chief_hosts is not None:
      self._is_master = (task_type == "chief")
    elif worker_hosts is not None:
      self._is_master = (task_type == "worker" and task_index == 0)
    else:
      raise RuntimeError("Not found chief or worker hosts in TF_CONFIG.")

    self._task_name = task_type
    self._task_index = task_index
    self._cluster_spec = cluster

  def _try_get_from_launch_tf_parameter(self):
    """Try get cluster config from launch_tf_param file on fuxi."""
    parameter_file = "{}/../launch_tf_param".format(os.getcwd())
    if not os.path.exists(parameter_file):
      logger.info("Not exist {} file.".format(parameter_file))
      return False
    with open(parameter_file, "r") as f:  # pylint: disable=unspecified-encoding
      parameter = f.read()
    import argparse  # pylint: disable=import-outside-toplevel
    parser = argparse.ArgumentParser(
        description='Process launch tf parameters.')
    parser.add_argument('--job_name', type=str, default="")
    parser.add_argument('--task_index', type=int, default=-1)
    parser.add_argument('--chief_hosts', type=str, default="")
    flags, _ = parser.parse_known_args(parameter.split())
    if not flags.job_name or flags.task_index == -1:
      raise ValueError("Invalid launch_tf_param {}".format(parameter))
    self._task_name = flags.job_name
    self._task_index = flags.task_index
    if not flags.chief_hosts:
      self._is_master = (self._task_name ==
                         "worker" and self._task_index == 0)
    else:
      self._is_master = (self._task_name == "chief")
    return True


def get_device_filters():
  """Get device dilters for PS job, device_filters can restrict how workers and PS can communicate.
     This can speed up training and ensure clean shutdowns in some situations.
  """
  distribute_config = DistributeConfig()
  task_name = distribute_config.task_name.lower()
  task_index = distribute_config.task_index
  device_filters = None
  if task_name == 'chief':
    device_filters = ['/job:ps', '/job:chief']
  elif task_name == 'worker':
    device_filters = ['/job:ps', '/job:worker/task:%d' % task_index]
  elif task_name == 'ps':
    device_filters = ['/job:ps', '/job:worker', '/job:chief']
  logger.info("Sparse cluster device filters {}".format(device_filters))

  return device_filters
