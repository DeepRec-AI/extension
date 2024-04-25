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
"""Global slice queue for training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from deeprec_master.proto.data_distributor import data_config_pb2 as config_pb
from deeprec_master.python.common.distribute_config import DistributeConfig
from deeprec_master.python.data.common.data_accessor import DataAccessor
from deeprec_master.python.data.common.exception import OutOfRangeException
from deeprec_master.python.data.training.config import InputDataConfig
from deeprec_master.python.utils.logger import logger

class GlobalSliceQueue(DataAccessor):
    """A global slice queue shared by all workers."""
    def __init__(self, filesource, num_epochs=1, shuffle=True, slice_size=10240,
                 ckpt_dir=None, name=None):
        self._cluster_config = DistributeConfig()
        self._data_config = \
            InputDataConfig(filesource, num_epochs, shuffle, slice_size)
        self._is_chief = self._cluster_config.is_chief
        self._ckpt_dir = ckpt_dir
        self._queue_name = "GlobalSliceQueue" if not name else name
        self._data_config_pb = self._gen_data_config_pb()
        # pylint: disable=super-with-arguments
        super(GlobalSliceQueue, self).__init__(
            task_name=self._cluster_config.task_name,
            task_index=self._cluster_config.task_index,
            data_config_pb=self._data_config_pb,
            is_chief=self._is_chief,
            ckpt_dir=self._ckpt_dir)

    def get_next_slice(self):
        """Get next slice."""
        while True:
            try:
                if self._data_config.dstype == \
                   InputDataConfig.DatasourceType.POSIX_FILE:
                    new_slice = self._get_next_posix_file_slice()
                else:
                    raise ValueError("Invalid datasource type.")
            except OutOfRangeException:
                logger.info("No more slice to get.")
                break
            for value in new_slice:
                yield value

    def _get_next_posix_file_slice(self):
        """Get file data slice from data manager.

        Returns:
        A filepath list will be returned for posix file,
        e.g. [path1, path2, ..., path1024];

        Raises:
        deeprecmaster.data.OutOfRangeException: if no more slice to get.
        """
        new_slice = self.get_slice_from_data_manager()
        slice_tag = new_slice.slice_tag
        slice_prefix = new_slice.slice_prefix
        slice_size = new_slice.slice_size
        slice_data = new_slice.slice_data
        assert slice_size == len(slice_data), \
            "Invalid new slice, slize size {} != {}".format(
                slice_size, len(slice_data))
        if slice_prefix is not None and len(slice_prefix) > 0:
            slice_prefix = slice_prefix.rstrip("/ ")
            slice_data = ["{}/{}".format(slice_prefix, data) \
                          for data in slice_data]
        logger.info("Get new slice, slice_id {}, slice_size {}".format(
            slice_tag, slice_size))
        return slice_data

    def _gen_data_config_pb(self):
      """Generate data pb config"""
      data_config = config_pb.DataConfig()
      data_config.config_name = self._queue_name
      data_config.consumer = config_pb.DataConfig.DataConsumerType.TRAINING

      if self._data_config.dstype == InputDataConfig.DatasourceType.POSIX_FILE:
        data_config.dstype = config_pb.DataConfig.DataSourceType.POSIX_FILE
        pf_config = config_pb.PosixFileConfig()
        for input_dir in self._data_config.filesource:
          pf_config.input_dirs.append(input_dir)
        pf_config.slice_size = self._data_config.slice_size
        data_config.file_config.CopyFrom(pf_config)

      config_options = config_pb.DataConfigOptions()
      config_options.shuffle = self._data_config.shuffle
      config_options.num_epochs = self._data_config.num_epochs
      data_config.options.CopyFrom(config_options)
      logger.info(data_config)
      return data_config
