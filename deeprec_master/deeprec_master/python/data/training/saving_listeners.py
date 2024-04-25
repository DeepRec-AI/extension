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
"""Saving Listeners for training."""

from tensorflow.python.training import basic_session_run_hooks
from deeprec_master.python.data.common.data_accessor import DataAccessor

class GlobalSliceQueueListener(basic_session_run_hooks.CheckpointSaverListener):
  """The listener that synchronize the generation of data ckpt and model ckpt.
  """
  def __init__(self, global_slice_queue):
    if not isinstance(global_slice_queue, DataAccessor):
      raise TypeError(
        'global_slice_queue must be a DataAccessor, given: {}'.format(
            global_slice_queue))
    self._global_slice_queue = global_slice_queue
    super().__init__()

  def before_save(self, session, global_step_value):
    self._global_slice_queue.stop_data_dispatch_and_save_ckpt()

  def after_save(self, session, global_step_value):
    self._global_slice_queue.start_data_dispatch()
    return False
