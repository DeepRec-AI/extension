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
"""AIMaster server process"""

import os
import time
import copy
import multiprocessing as mp
from deeprec_master.python.utils.logger import logger
from deeprec_master.python.scheduler import Scheduler
from deeprec_master.python.metrics_collector.metrics_collector import MetricSuber

class AIMasterServerProcess(mp.Process):
  """AIMaster server process."""
  def __init__(self, aimaster_ip, aimaster_port):
    super(AIMasterServerProcess, self).__init__(name="AIMasterServerProcess") # pylint: disable=super-with-arguments
    self._aimaster_ip = aimaster_ip
    self._aimaster_port = aimaster_port
    self._job_metrics = mp.Manager().list()

  def run(self):
    """Run AIMaster server process."""
    logger.info("AIMasterServerProcess pid: {}".format(os.getpid()))
    aimaster_rpc_service = Scheduler(self._aimaster_ip, self._aimaster_port)
    logger.info('AIMaster service started at: {}'.format(aimaster_rpc_service.addr))
    logger.info('AIMaster service running ...')
    self._suber = MetricSuber('suber', ['loss', 'cpu', 'gpu'])

    while True:
      for metric_name in ['gpu', 'cpu', 'loss']:
        metric = self._suber.get_metric(metric_name)
        if metric:
          logger.info("New job metric: {}".format(metric))
          self._job_metrics.append(copy.deepcopy(metric))
      time.sleep(1)

  def get_job_metrics(self):
    """Return job metrics."""
    return self._job_metrics
