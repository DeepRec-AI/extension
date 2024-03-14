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

"""Test for elastic partition op."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dynamic_embedding_server.python import elastic_training_hooks

import os
import json
import portpicker

from tensorflow.python.platform import test
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import embedding_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.layers import layers
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.framework import ops
from tensorflow.python.ops import kv_variable_ops
from tensorflow.python.training import training_util
from tensorflow.python.training import adagrad
from tensorflow.python.training import server_lib
from tensorflow.python.training import device_setter
from tensorflow.python.training import training
from tensorflow.python.training.monitored_session import MonitoredTrainingSession

def create_local_cluster(num_workers, num_ps, protocol="grpc"):
  """Create local GRPC servers and return them."""
  worker_ports = [portpicker.pick_unused_port() for _ in range(num_workers)]
  ps_ports = [portpicker.pick_unused_port() for _ in range(num_ps)]
  cluster_dict = {
      "worker": ["localhost:%s" % port for port in worker_ports],
      "ps": ["localhost:%s" % port for port in ps_ports]
  }
  cs = server_lib.ClusterSpec(cluster_dict)

  workers = [
      server_lib.Server(
          cs, job_name="worker", protocol=protocol, task_index=ix, start=True)
      for ix in range(num_workers)
  ]
  ps_servers = [
      server_lib.Server(
          cs, job_name="ps", protocol=protocol, task_index=ix, start=True)
      for ix in range(num_ps)
  ]

  return cluster_dict, workers, ps_servers

# Creates the workers and return their sessions, graphs, train_ops.
# Chief worker will update at last
def _get_workers(num_workers, workers, num_ps=1):
  sessions = []
  graphs = []
  train_ops = []
  losses = []
  for worker_id in range(num_workers):
    graph = ops.Graph()
    is_chief = (worker_id == 0)
    with graph.as_default():
      worker_device = "/job:worker/task:%d/cpu:0" % (worker_id)
      with variable_scope.variable_scope(
          ""), ops.device(
              device_setter.replica_device_setter(
                  worker_device=worker_device,
                  ps_device="/job:ps/task:0/cpu:0",
                  ps_tasks=1)):
        global_step = training_util.get_or_create_global_step()
      if num_ps > 1:
        with variable_scope.variable_scope(
            "",
            partitioner=partitioned_variables.fixed_size_partitioner(
                num_ps, axis=0)), ops.device(
                device_setter.replica_device_setter(
                    worker_device=worker_device,
                    ps_device="/job:ps/task:0/cpu:0",
                    ps_tasks=num_ps)):

          var_0 = variable_scope.get_embedding_variable("var_0",
                                        embedding_dim=16,
                                        initializer=init_ops.ones_initializer(dtypes.float32),
                                        # partitioner=partitioner,
                                        ev_option = variables.EmbeddingVariableOption(filter_option=None))

      with ops.device("/job:worker/task:" + str(worker_id)):
        emb = embedding_ops.embedding_lookup(var_0, math_ops.cast([0,1,2,3,4,5], dtypes.int64))
        fun = math_ops.multiply(emb, 2.0, name='multiply')
        dnn_input = layers.dense(fun,
                                    units=12,
                                    activation=nn.relu)
        dnn_input = layers.batch_normalization(
                            dnn_input, training=True, trainable=True)
        loss = math_ops.reduce_sum(dnn_input, name='reduce_sum')
        opt = adagrad.AdagradOptimizer(0.1)
        g_v = opt.compute_gradients(loss)
        train_op = opt.apply_gradients(g_v, global_step)
      # Creates MonitoredSession
      sess = training.MonitoredTrainingSession(
          workers[worker_id].target, hooks=[elastic_training_hooks.ElasticTrainingHook()])

    sessions.append(sess)
    graphs.append(graph)
    train_ops.append(train_op)
    losses.append(loss)

  return sessions, graphs, train_ops, losses

class ElasticTrainingHooksTest(test.TestCase):
    def testScalingDown(self):
        num_workers = 2
        num_ps = 2
        cluster, workers, _ = create_local_cluster(
            num_workers=num_workers, num_ps=num_ps)

        sessions, graphs, train_ops, losses = _get_workers(num_workers, workers, 2)
            
        for i in range(1100):
            print(sessions[0].run([train_ops[0], losses[0]]))

if __name__ == "__main__":
    os.environ["TF_CONFIG"] = json.dumps({
        "cluster": {
            "worker": ["localhost:20086", "localhost:20087"],
            "ps": ["localhost:10086", "localhost:10087"]
        },
    "task": {"type": "worker", "index": 0}
    })
    test.main()