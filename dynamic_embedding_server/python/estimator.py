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
"""DES public interface"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.distribute import estimator_training as distribute_coordinator_training
from tensorflow_estimator.python.estimator import run_config as run_config_lib
from tensorflow.python.estimator import training as orig_training
ENABLE_DYNAMIC_PS = "ENABLE_DES"
def is_enable_elastic_ps():
    enable_des = os.environ.get(ENABLE_DYNAMIC_PS, False)
    return enable_des
def _assert_eval_spec(eval_spec):
    """Raise error if `eval_spec` is not of the right type."""
    if not isinstance(eval_spec, orig_training.EvalSpec):
        raise TypeError('`eval_spec` must have type `tf.estimator.EvalSpec`. '
                        'Got: {}'.format(type(eval_spec)))
class _TrainingExecutorForDES(orig_training._TrainingExecutor):
    def __init__(self,
                 estimator,
                 train_spec,
                 eval_spec,
                 train_hooks=None):
        is_des_enabled = is_enable_elastic_ps()
        if is_des_enabled:
            logging.info("Running estimator with DyanmicEmbeddingServer enabled.")
            from dynamic_embedding_server.python.elastic_training_hooks import ElasticTrainingHook
            prev_protocol = estimator.config.protocol
            logging.info("Changing protocol from {} to {}.".format(prev_protocol, "elastic-grpc"))
            estimator.config = run_config_lib.RunConfig.replace(
                estimator.config, protocol="elastic-grpc")
            if train_hooks is None:
                hooks = [ElasticTrainingHook()]
            else:
                if not any(isinstance(x, ElasticTrainingHook) for x in hooks):
                    hooks.append(ElasticTrainingHook())
        super().__init__(estimator, train_spec, eval_spec, train_hooks)
def train_and_evaluate(estimator, train_spec, eval_spec):
    _assert_eval_spec(eval_spec)  # fail fast if eval_spec is invalid.
    executor = _TrainingExecutorForDES(
        estimator=estimator, train_spec=train_spec, eval_spec=eval_spec)
    config = estimator.config
    # If `distribute_coordinator_mode` is set and running in distributed
    # environment, we run `train_and_evaluate` via distribute coordinator.
    if distribute_coordinator_training.should_run_distribute_coordinator(config):
        logging.info('Running `train_and_evaluate` with Distribute Coordinator.')
        distribute_coordinator_training.train_and_evaluate(
            estimator, train_spec, eval_spec, _TrainingExecutorForDES)
        return
    if (config.task_type == run_config_lib.TaskType.EVALUATOR and
        config.task_id > 0):
        raise ValueError(
            'For distributed training, there can only be one `evaluator` task '
            '(with task id 0).  Given task id {}'.format(config.task_id))
    return executor.run()
