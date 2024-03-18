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
"""Common retry module."""

import time


class RetryCaller:
    """Retry caller"""

    def __init__(
        self, max_retry=3, base_delay=1, max_delay=10, exceptions=(Exception,)
    ):
        self._max_retry_count = max_retry
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._exceptions = exceptions

    def __call__(self, func, *args, **kwargs):
        for retry_count in range(self._max_retry_count + 1):
            try:
                return func(*args, **kwargs)
            except self._exceptions as e:
                if retry_count == self._max_retry_count:
                    raise e
            self._retry_wait(retry_count)
        return None

    def _retry_wait(self, retry_count):
        delay = min(self._base_delay * 2**retry_count, self._max_delay)
        time.sleep(delay)
