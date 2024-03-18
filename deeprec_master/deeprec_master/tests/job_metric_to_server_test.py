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
"""Test for job metric to master."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import time
import os

from deeprec_master.python.utils.network import get_random_port

class JobMetricToServerTest(unittest.TestCase):
  """Test job metric to server."""

  aimaster_server = ""
  aimaster_addr = ""

  @classmethod
  def setUpClass(cls):
    from scheduler_test import AIMasterServerProcess # pylint: disable=import-outside-toplevel
    aimaster_ip = "127.0.0.1"
    aimaster_port = get_random_port()
    JobMetricToServerTest.aimaster_addr = aimaster_ip + ":" + str(aimaster_port)
    aimaster_server = AIMasterServerProcess(aimaster_ip, aimaster_port)
    aimaster_server.start()
    time.sleep(3)
    JobMetricToServerTest.aimaster_server = aimaster_server

  @classmethod
  def tearDownClass(cls):
    JobMetricToServerTest.aimaster_server.terminate()

  def test_send_and_analyze_metric(self):
    """Test sending and analyzing job metric."""
    os.environ['DEEPRECMASTER_ADDR'] = JobMetricToServerTest.aimaster_addr

    # For job worker, sending job metric to aimaster server
    # jm_config = jm.TFConfig("worker", 0)
    # monitor = jm.Monitor(config=jm_config)
    # metric_options = {"gpu_type": "A100"}
    # monitor.report("gpu", 0.56, **metric_options)
    # monitor.report("cpu", 220)
    # monitor.report("loss", 0.12)

    time.sleep(5)

    # For AIMaster server, analyzing job metrics
    job_metrics = JobMetricToServerTest.aimaster_server.get_job_metrics()
    self.assertEqual(len(job_metrics), 3)
    for metric in job_metrics:
      if metric.type == 'loss':
        self.assertEqual(metric.value, 0.12)
      elif metric.type == 'gpu':
        self.assertEqual(metric.value, 0.56)
      elif metric.type == 'cpu':
        self.assertEqual(metric.value, 220)

if __name__ == '__main__':
  unittest.main()
