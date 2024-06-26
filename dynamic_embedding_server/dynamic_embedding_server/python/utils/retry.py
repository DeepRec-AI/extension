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

import functools

def retry(retries=3):
    def retry_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            remaining_attempts = retries
            while remaining_attempts > 1:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"func {func.__name__} exec failed,is retrying and remaining count is: {remaining_attempts - 1}")
                    remaining_attempts -= 1
            return func(*args, **kwargs)
        return wrapper
    return retry_decorator
