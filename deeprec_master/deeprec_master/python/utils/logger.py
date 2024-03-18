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
""" logger of deeprecmaster """

import logging
import os


class LogConst:
    AIMASTER_LIB_PATH = "/home/pai/lib/python3.6/site-packages/"
    AIMASTER_LIB_PATH_CONDA = "/opt/conda/envs/aimaster/lib/python3.6/site-packages/"


class LogFormatter(logging.Formatter):
    """Log formatter class."""

    def format(self, record):
        if "pathname" in record.__dict__.keys():
            if record.pathname.startswith(LogConst.AIMASTER_LIB_PATH):
                record.pathname = record.pathname[len(LogConst.AIMASTER_LIB_PATH) :]
            elif record.pathname.startswith(LogConst.AIMASTER_LIB_PATH_CONDA):
                record.pathname = record.pathname[
                    len(LogConst.AIMASTER_LIB_PATH_CONDA) :
                ]
            else:
                _, filename = os.path.split(record.pathname)
                record.pathname = filename
            record.pathname = os.path.basename(record.pathname)
        return super(LogFormatter, self).format(
            record
        )  # pylint: disable=super-with-arguments


class Logger:
    """logger class of deeprec_master"""

    level_relations = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "crit": logging.CRITICAL,
    }

    def __init__(self, modulename):
        self.level = "info"
        self.logger = logging.getLogger(modulename)
        self.logger.propagate = False
        self.logger.setLevel(self.level_relations.get(self.level))
        self.fmt = "[%(asctime)s DeepRecMaster] (%(pathname)s %(lineno)d) %(levelname)s: %(message)s"
        self.format_str = LogFormatter(self.fmt)
        sh = logging.StreamHandler()
        sh.setLevel(level=logging.INFO)
        sh.setFormatter(self.format_str)
        self.logger.addHandler(sh)


logger = Logger("deeprecmaster").logger
