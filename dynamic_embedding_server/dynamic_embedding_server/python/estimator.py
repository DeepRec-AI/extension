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
"""DES public interface"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from dynamic_embedding_server.python.utils.server_utils import estimator_prepare_or_wait_for_cluster

from tensorflow.python.estimator import run_config as run_config_lib
from tensorflow.python.estimator import training as orig_training
from dynamic_embedding_server.python.utils.logger import logger
from tensorflow.python.distribute import estimator_training as distribute_coordinator_training
from tensorflow.python.estimator import estimator as orig_estimator


def _assert_eval_spec(eval_spec):
  """Raise error if `eval_spec` is not of the right type."""
  if not isinstance(eval_spec, orig_training.EvalSpec):
    raise TypeError('`eval_spec` must have type `tf.estimator.EvalSpec`. '
                    'Got: {}'.format(type(eval_spec)))


class _TrainingExecutorForDES(orig_training._TrainingExecutor):

  def __init__(self, estimator, train_spec, eval_spec, train_hooks=None):
    logger.info("Running estimator with DyanmicEmbeddingServer enabled.")
    from dynamic_embedding_server.python.elastic_training_hooks import ElasticTrainingHook
    prev_protocol = estimator.config.protocol
    logger.info("Changing protocol from {} to {}.".format(
        prev_protocol, "elastic-grpc"))
    estimator._config = run_config_lib.RunConfig.replace(
        estimator.config, protocol="elastic-grpc")
    if train_hooks is None:
      train_hooks = [ElasticTrainingHook()]
    else:
      if not any(
              isinstance(x, ElasticTrainingHook) for x in train_hooks):
        train_hooks.append(ElasticTrainingHook())
    super(_TrainingExecutorForDES, self).__init__(estimator, train_spec,
                                                  eval_spec, train_hooks)

  def _start_std_server(self, config):
    from tensorflow.python.training import server_lib
    cluster_spec = estimator_prepare_or_wait_for_cluster(config)
    config._cluster_spec = server_lib.ClusterSpec(cluster_spec)
    self._estimator._config._cluster_spec = config._cluster_spec
    self._estimator._config._num_ps_replicas = run_config_lib._count_ps(
        config._cluster_spec)
    self._estimator._device_fn = (
        self._estimator._config.device_fn or
        orig_estimator._get_replica_device_setter(self._estimator._config))
    logger.info("dynamic ps estimate done, new clusterspec:%s, %s",
                 cluster_spec, self._estimator._config)
    return super(_TrainingExecutorForDES, self)._start_std_server(config)


def train_and_evaluate(estimator, train_spec, eval_spec):
  _assert_eval_spec(eval_spec)  # fail fast if eval_spec is invalid.
  executor = _TrainingExecutorForDES(estimator=estimator,
                                     train_spec=train_spec,
                                     eval_spec=eval_spec,
                                     train_hooks=train_spec.hooks)
  config = estimator.config
  # If `distribute_coordinator_mode` is set and running in distributed
  # environment, we run `train_and_evaluate` via distribute coordinator.
  if distribute_coordinator_training.should_run_distribute_coordinator(
          config):
    logger.info(
        'Running `train_and_evaluate` with Distribute Coordinator.')
    distribute_coordinator_training.train_and_evaluate(
        estimator, train_spec, eval_spec, _TrainingExecutorForDES)
    return
  if (config.task_type == run_config_lib.TaskType.EVALUATOR
          and config.task_id > 0):
    raise ValueError(
        'For distributed training, there can only be one `evaluator` task '
        '(with task id 0).  Given task id {}'.format(config.task_id))
  return executor.run()
