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
"""Data accessor for accessing deeprecmaster server data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import hashlib
import os
import tensorflow as tf
import time
from google.protobuf import text_format

from deeprec_master import pywrap_deeprecmaster
from deeprec_master.proto.data_distributor import data_config_pb2
from deeprec_master.python.data.common.exception import OutOfRangeException
from deeprec_master.python.utils import util_func
from deeprec_master.python.utils.logger import logger

class AccessorConstants:
    CKPT_FILE_NAME = "data_consumption_ckpt.pbtxt"

class DataAccessor(object): # pylint: disable=useless-object-inheritance
    """Data accessor"""
    def __init__(self,
                 task_name,
                 task_index,
                 data_config_pb,
                 is_chief=None,
                 ckpt_dir=None):
        self._task_name = task_name
        self._task_index = task_index
        self._data_config_pb = data_config_pb
        self._is_chief = is_chief
        self._ckpt_dir = ckpt_dir
        self._deeprecmaster_addr = os.getenv('DEEPRECMASTER_ADDR')
        if self._deeprecmaster_addr is None:
            raise RuntimeError("Not found DeepRecMaster address.")
        self._data_manager_is_ready = False

    def initialize_deeprecmaster_data_manager(self):
        """Initialize deeprecmaster data manager."""
        logger.info("Initializing deeprecmaster data manager.")
        self._try_restore_from_ckpt()
        self._data_config_bytes = self._data_config_pb.SerializeToString()
        self._init_data_manager()
        self._waiting_for_data_manager_ready()

    def _try_restore_from_ckpt(self):
        """Try restoring data state from ckpt."""
        if not self._is_chief or not self._ckpt_dir:
            return False
        if not tf.gfile.IsDirectory(self._ckpt_dir):
            raise ValueError("{} not a directory.".format(self._ckpt_dir))
        self._ckpt_file_path = "{}/{}".format(
            self._ckpt_dir.rstrip('/ '), AccessorConstants.CKPT_FILE_NAME)
        if not tf.gfile.Exists(self._ckpt_file_path):
            logger.info("{} not exists.".format(self._ckpt_file_path))
            return None
        logger.info("Try restoring elastic data state from {}".format(
            self._ckpt_file_path))
        data_config_ckpt = data_config_pb2.DataConfig()
        with tf.gfile.GFile(self._ckpt_file_path, mode='r') as fin:
            proto_str = fin.read()
            text_format.Merge(proto_str, data_config_ckpt)
        new_config_name = self._data_config_pb.config_name
        if new_config_name == data_config_ckpt.config_name:
            self._data_config_pb = data_config_ckpt
        return True

    def _init_data_manager(self):
        """init data manager"""
        if self._is_chief is not None and not self._is_chief:
            return False
        max_retry_count = 3
        for _ in range(max_retry_count):
            st = pywrap_deeprecmaster.init_data_manager(
                self._deeprecmaster_addr, self._task_name, self._task_index,
                self._data_config_bytes)
            if st.ok():
                logger.info("Init deeprecmaster data manager success.")
                return True
            time.sleep(2)
        raise RuntimeError(
            "Failed to init deeprecmaster data manager, {}".format(
                st.to_string()))

    @util_func.func_run_time_printer(60)
    def _waiting_for_data_manager_ready(self):
        """Waiting for data manager ready."""
        logger.info("Waiting for deeprecmaster data manager ready.")
        while not self._check_data_manager_is_ready():
            time.sleep(5)
        self._data_manager_is_ready = True

    def _check_data_manager_is_ready(self):
        """Return if data manager is ready."""
        max_retry_count = 3
        for _ in range(max_retry_count):
            result = pywrap_deeprecmaster.is_ready_data_manager(
                self._deeprecmaster_addr, self._task_name, self._task_index)
            if result.status.ok():
                return result.is_ready
            time.sleep(2)
        raise RuntimeError("Failed to get deeprecmaster status, {}".format(
            result.status.to_string()))

    def get_slice_from_data_manager(self):
        """Get slice from data manager."""
        max_retry_count = 3
        retry_count = 0
        while retry_count < max_retry_count:
            # pylint: disable=c-extension-no-member
            result = pywrap_deeprecmaster.get_slice_from_data_manager(
                self._deeprecmaster_addr, self._task_name, self._task_index)
            if result.status.ok():
                return result
            if result.status.code() == \
               pywrap_deeprecmaster.ErrorCode.RETRY_LATER:
                time.sleep(2)
                continue
            if result.status.code() == \
               pywrap_deeprecmaster.ErrorCode.OUT_OF_RANGE:
                error_msg = "No more slice to get."
                logger.warning(error_msg)
                raise OutOfRangeException(error_msg)
            retry_count += 1
            time.sleep(2)
        raise RuntimeError("Failed to get slice. {}".format(
            result.status.to_string()))

    def start_data_dispatch(self):
        """enable data manager to dispatch data."""
        max_retry_count = 3
        for _ in range(max_retry_count):
            st = pywrap_deeprecmaster.start_data_dispatch(
                self._deeprecmaster_addr, self._task_name, self._task_index)
            if st.ok():
                logger.info("enable deeprecmaster to dispatch data success.")
                return
            raise RuntimeError(
                "Failed to enable deeprecmaster to dispatch data, {}".format(
                    st.to_string()))

    def stop_data_dispatch_and_save_ckpt(self):
        """Stop data manager dispatching data, and generate ckpt."""
        result = \
            self._stop_data_dispatch_and_get_data_state_from_deeprecmaster()
        if self._ckpt_dir is not None:
            self._save_ckpt_to_file(result)

    def _stop_data_dispatch_and_get_data_state_from_deeprecmaster(self):
        """
        Stop data manager dispatching data,and get data state from
        deeprecmaster.
        """
        max_retry_count = 3
        for _ in range(max_retry_count):
            result = pywrap_deeprecmaster.stop_data_dispatch_and_get_data_state(
                self._deeprecmaster_addr, self._task_name, self._task_index)
            if result.status.ok():
                return result
            time.sleep(2)
        raise RuntimeError(
            "Failed to stop data dispatch and get data state. {}".format(
                result.status.to_string()))

    def _save_ckpt_to_file(self, result, prev_data_state_md5=None):
        """save data state to file."""
        new_data_config = data_config_pb2.DataConfig()
        data_state = base64.b64decode(result.data_state)
        new_data_config.ParseFromString(data_state)
        new_data_state_md5 = \
            hashlib.md5(str(new_data_config).encode('utf-8')).hexdigest()
        if new_data_state_md5 != prev_data_state_md5:
            tmp_file_path = self._ckpt_file_path + '.tmp'
            with tf.gfile.GFile(tmp_file_path, "w") as gf:
                gf.write(str(new_data_config))
            tf.gfile.Rename(tmp_file_path, self._ckpt_file_path, overwrite=True)
        return new_data_state_md5
