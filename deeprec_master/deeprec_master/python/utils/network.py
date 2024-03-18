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
""" network related funcs """

import os
import socket

from deeprec_master.python.utils import constants
from deeprec_master.python.utils.logger import logger


def get_my_ip():
    return socket.gethostbyname(socket.gethostname())


def get_random_port(num=1):
    """return random ports"""
    ports = []
    sockets = []
    try:
        for _ in range(num):
            s = socket.socket()
            s.bind(("", 0))
            ports.append(s.getsockname()[1])
            sockets.append(s)
    finally:
        for s in sockets:
            s.close()
    if len(ports) == num:
        return ports[0] if num == 1 else ports
    raise ValueError("get ports failed")


def get_ip_and_port(check_env=True):
    """Get ip and port, get value from env if check_env enabled."""
    ip = socket.gethostbyname(socket.gethostname())
    port = 0  # can use a port number of 0 to assign a random port to a gRPC
    master_addr = os.getenv(constants.DEEPRECMASTER_ADDR_ENV)
    if check_env and master_addr:
        logger.info("{} Env: {}".format(constants.DEEPRECMASTER_ADDR_ENV, master_addr))
        addr_part = master_addr.split(":")
        if len(addr_part) != 2:
            raise RuntimeError("Invalid deeprecmaster host: {}".format(master_addr))
        port = int(addr_part[1])
    return ip, port
