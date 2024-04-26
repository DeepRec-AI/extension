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
"""Hooks for training."""

import time
import os

from deeprec_master import pywrap_deeprecmaster
from deeprec_master.python.common.distribute_config import DistributeConfig
from tensorflow.python.training import session_run_hook
from deeprec_master.python.utils.logger import logger

class ModelReadyHook(session_run_hook.SessionRunHook):
    """The hook for model parameter ready."""
    def __init__(self):
        self._cluster_config = DistributeConfig()
        self._is_chief = self._cluster_config.is_chief
        self._task_name = self._cluster_config.task_name
        self._task_index = self._cluster_config.task_index
        self._deeprecmaster_addr = os.getenv('DEEPRECMASTER_ADDR')
        if self._deeprecmaster_addr is None:
            raise RuntimeError("Not found DeepRecMaster address.")

    def before_create_session(self):
        if not self._is_chief:
            return
        self._set_model_ready_state(False)

    def after_create_session(self, session, coord):
        if self._is_chief:
            self._set_model_ready_state(True)
        else:
            self._waiting_for_model_ready()

    def _set_model_ready_state(self, ready_state):
        logger.info("Set model parameter ready state to {}".format(ready_state))
        pywrap_deeprecmaster.model_ready_mgr_set_state(
            self._deeprecmaster_addr, self._task_name, self._task_index,
            ready_state)

    def _waiting_for_model_ready(self):
        logger.info("Waiting for model parameter to be ready.")
        while not self._check_model_is_ready():
            time.sleep(5)

    def _check_model_is_ready(self):
        """Return if model parameter is ready."""
        max_retry_count = 3
        for _ in range(max_retry_count):
            result = pywrap_deeprecmaster.model_ready_mgr_get_state(
                self._deeprecmaster_addr, self._task_name, self._task_index)
            if result.status.ok():
                return result.ready_state
            time.sleep(2)
        raise RuntimeError("Failed to get model_ready_mgr status, {}".format(
            result.status.to_string()))
