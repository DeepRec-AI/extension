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
"""Timer."""

import time
from deeprec_master.python.utils.logger import logger


class Timer(object):  # pylint: disable=useless-object-inheritance
    """Base timer"""

    def __init__(self):
        """Init of timer"""

    def now(self):
        """Get current time"""
        raise NotImplementedError("must be implemented in descendants")


class RealTimer(Timer):
    """Timer that uses true time"""

    def __init__(self):  # pylint: disable=useless-super-delegation
        """Init of RealTimer"""
        super(RealTimer, self).__init__()  # pylint: disable=super-with-arguments

    def now(self):
        """Return current true time"""
        return int(time.time())


class MockTimer(Timer):
    """Timer for mock testing"""

    def __init__(self):
        """Init of mock timer"""
        super(MockTimer, self).__init__()  # pylint: disable=super-with-arguments
        logger.info("Mock timer created")
        self.now_time = None

    def set_now_time(self, cur_now_time):
        """Mannually change current time"""
        self.now_time = cur_now_time

    def now(self):
        """Return current true time"""
        return self.now_time
