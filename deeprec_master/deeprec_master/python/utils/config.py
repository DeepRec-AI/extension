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
"""Config for job monitor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import os
import json
import six


class JobMonitorDefaultConfig:
    """Job monitor default config."""
    ENABLE_CHIEF_FAILOVER = False
    ENABLE_PS_FAILOVER = False
    MAX_WAITING_TIME_FOR_CHIEF_PS_FAILOVER = 5 * 60  # 5 min
    ENABLE_DYNAMIC_EMBEDDING_SERVER = False


class JobMonitorConfig:
    """Job monitor config class."""

    def __init__(self, args=None):
        default_configs = JobMonitorConfig.get_default_configs()
        for key, value in default_configs.items():
            self.__dict__[key] = value if args is None else getattr(args, key)

    @staticmethod
    def get_default_configs():
        """Get default configs."""
        configs = {
            "enable_chief_failover": JobMonitorDefaultConfig.ENABLE_CHIEF_FAILOVER,
            "enable_ps_failover": JobMonitorDefaultConfig.ENABLE_PS_FAILOVER,
            "max_waiting_time_for_chief_ps_failover": (
                JobMonitorDefaultConfig.MAX_WAITING_TIME_FOR_CHIEF_PS_FAILOVER
            ),
            "enable_dynamic_embedding_server": JobMonitorDefaultConfig.ENABLE_DYNAMIC_EMBEDDING_SERVER,
        }
        return configs

    def to_json(self):
        """Serialize object to json."""
        return json.dumps(self.__dict__)
