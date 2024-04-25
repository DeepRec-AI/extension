# Copyright 2024 DeepRec Authors. All Rights Reserved.
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
"""Deeprecmaster distributed job configs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

from deeprec_master.python.utils.logger import logger

class DistributeConfig(object):
    """deeprecaster distributed job configs."""
    def __init__(self):
        self._get_tfjob_cluster_config()
        logger.info("task_name:{}, task_index:{}, is_chief:{}".format(
            self._task_name, self._task_index, self._is_chief))

    @property
    def task_name(self):
        return self._task_name

    @property
    def task_index(self):
        return self._task_index

    @property
    def is_chief(self):
        return self._is_chief

    def _get_tfjob_cluster_config(self):
        """Get tf job cluster config."""
        tf_config_str = os.getenv('TF_CONFIG')
        if tf_config_str is None:
            raise RuntimeError("Not found `TF_CONFIG` env variable.")

        logger.info("TF_CONFIG {}".format(tf_config_str))
        try:
            tf_config = json.loads(tf_config_str)
        except Exception as ex:
            raise ValueError(
                "Invalid json format TF_CONFIG:{}, exception:{}".format(
                    tf_config_str, ex))
        assert 'cluster' in tf_config and 'task' in tf_config, \
            "Invalid TF_CONFIG: {}, no cluster or task key.".format(tf_config)

        task = tf_config['task']
        assert 'type' in task and 'index' in task, \
        "Invalid TF_CONFIG task: {}, no type or index key.".format(tf_config)
        task_type = task['type']
        task_index = int(task['index'])

        cluster = tf_config['cluster']
        chief_hosts = cluster.get('chief', None)
        worker_hosts = cluster.get('worker', None)
        if chief_hosts is not None:
            self._is_chief = (task_type == "chief")
        elif worker_hosts is not None:
            self._is_chief = (task_type == "worker" and task_index == 0)
        else:
            raise RuntimeError("Not found chief or worker hosts in TF_CONFIG.")
        self._task_name = task_type
        self._task_index = task_index
