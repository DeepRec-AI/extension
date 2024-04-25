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
"""Config for training data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from enum import Enum
from six import string_types

from deeprec_master.python.utils.logger import logger

class InputDataConfig(object):
    """Input data config"""
    class DatasourceType(Enum):
        POSIX_FILE = 0

    def __init__(self, filesource, num_epochs=1, shuffle=True,
                 slice_size=10240):
        self._filesource = filesource
        self._num_epochs = num_epochs
        self._shuffle = shuffle
        self._slice_size = slice_size
        self._validate()
        self._dstype = self._get_datasource_type()
        logger.info(
            "filesource:{}, num_epochs:{}, shuffle:{}, slice_size:{}".format(
                self._filesource, self._num_epochs, self._shuffle,
                self._slice_size))

    @property
    def filesource(self):
        return self._filesource

    @property
    def num_epochs(self):
        return self._num_epochs

    @property
    def slice_size(self):
        return self._slice_size

    @property
    def shuffle(self):
        return self._shuffle

    @property
    def dstype(self):
        return self._dstype

    def _validate(self):
        """Validate config values."""
        if not isinstance(self._filesource, list) or not self._filesource:
            raise ValueError(
                "filesource should be a list of strings")
        self._filesource = [fs.encode() if isinstance(fs, string_types)
                             else fs for fs in self._filesource]
        if not all(isinstance(fs, bytes) for fs in self._filesource):
            raise ValueError(
                "ilesource should be a list of strings not {}".format(
                    [type(fs) for fs in self._filesource]))
        self._filesource = \
            [str(fs.strip().decode()) for fs in self._filesource]
        assert self._num_epochs > 0, \
            "num_epochs must be > 0 not {}".format(self._num_epochs)
        assert isinstance(self._shuffle, bool), \
            "shuffle should be bool type, not {}".format(type(self._shuffle))
        assert self._slice_size > 0, \
            "slice_size must be > 0 not {}".format(self._slice_size)

    def _get_datasource_type(self):
        if all(fs.startswith("/") for fs in self._filesource):
            return InputDataConfig.DatasourceType.POSIX_FILE

        raise ValueError("Not found available filesource " \
                         "type from {}".format(self._filesource))
