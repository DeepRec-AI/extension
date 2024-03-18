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
from __future__ import absolute_import, division, print_function

import time

from deeprec_master import pywrap_deeprecmaster as pywrap
from deeprec_master.python.utils.logger import logger


class Scheduler:
    "Scheduler to gather service"
    _service = None

    def __init__(self, ip, port):
        """ Initialize a Scheduler object """
        self._service = pywrap.new_scheduler(ip, port)
        self._addr = self._service.start()

    @property
    def addr(self):
        return self._addr

    def join(self):
        """ Join service """
        self._service.join()

    def stop(self):
        """ Stop service """
        if self._service is not None:
            self._service = None

    def __del__(self):
        """ clean when deleting the object """
        self.stop()
