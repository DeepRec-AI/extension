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
""" logger of DynamicEmbeddingServer """

# pylint: disable=W1505

import logging
import os

class Logger():
  """ logger class of aimaster """
  level_relations = {
    'debug':logging.DEBUG,
    'info':logging.INFO,
    'warning':logging.WARNING,
    'error':logging.ERROR,
    'crit':logging.CRITICAL
  }

  def __init__(self, modulename):
    self.level = 'info'
    self.logger = logging.getLogger(modulename)
    self.logger.propagate = False
    self.logger.setLevel(self.level_relations.get(self.level))
    self.fmt='[%(asctime)s DynamicEmbeddingServer] (%(pathname)s %(lineno)d) %(levelname)s: %(message)s'
    self.format_str = logging.Formatter(self.fmt)
    sh = logging.StreamHandler()
    sh.setLevel(level=logging.INFO)
    sh.setFormatter(self.format_str)
    self.logger.addHandler(sh)

logger = Logger('DynamicEmbeddingServer').logger
