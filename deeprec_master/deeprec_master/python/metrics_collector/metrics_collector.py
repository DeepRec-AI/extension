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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import deepcopy
import fnmatch
import json
import time
import threading
from threading import Thread
import sys

from deeprec_master.python.utils.logger import logger
from deeprec_master.python.utils.timer import RealTimer
from deeprec_master import pywrap_deeprecmaster

if sys.version_info.major == 2:
    import Queue as queue
else:
    import queue


NODE_METRICS = ["cpu_mem", "cpu_util_usage"]
GRAPH_METRICS = ["global_step", "resource_stat"]


class Metric:
    """Metric class."""

    def __init__(self, type_name, timestamp, value, **kwargs):
        self.type = type_name
        self.timestamp = timestamp
        self.value = value
        self.options = kwargs

    def __str__(self):
        return "type:{}, timestamp:{}, value:{}, options:{}".format(
            self.type, self.timestamp, self.value, self.options
        )


class MetricsCollector:
    """Metric manager."""
    #TODO(JUNQI): consider to be adjustable
    WINDOW_SIZE = 10
    def __init__(self):
        self._subscribers = set()
        self._lock = threading.Lock()
        self._metrics_mgr = pywrap_deeprecmaster.get_metrics_mgr()
        self._exit_metric_handler = False
        self.run_gazer_handle()

    def __del__(self):
        self.set_metric_handler_exit()
        while not self._metric_handler_thread.is_alive():
            time.sleep(2)
            logger.info(
                "{} thread alive, wait exit...".format(
                    self._metric_handler_thread.name)
            )

    def add_suber(self, suber):
        """Add subscriber."""
        if not isinstance(suber, MetricSuber):
            logger.warning("{} not MetricSuber object.".format(suber))
            return False
        with self._lock:
            if suber in self._subscribers:
                return False
            self._subscribers.add(suber)
        logger.info("Subscriber {} attached.".format(suber.name))
        return True

    def remove_suber(self, suber):
        """Remove subscriber."""
        if not isinstance(suber, MetricSuber):
            logger.warning("{} not MetricSuber object.".format(suber))
            return False
        with self._lock:
            if suber not in self._subscribers:
                return False
            self._subscribers.discard(suber)
        logger.info("Subscriber {} removed.".format(suber.name))
        return True

    def set_metric_handler_exit(self):
        """Set metric handler exit."""
        self._exit_metric_handler = True

    def run_gazer_handle(self):
        """Run metric handler."""

        def _handle_thread_func():
            while not self._exit_metric_handler:
                time.sleep(MetricsCollector.WINDOW_SIZE)
                with self._lock:
                    metrics = self._metrics_mgr.get_all_actions()
                    self.process_metrics(metrics)
            logger.info("{} exit.".format(self._metric_handler_thread.name))

        self._gazer_handle_thread = threading.Thread(
            target=_handle_thread_func, name="gazer_handle_thread", daemon=True
        )
        self._gazer_handle_thread.start()
        logger.info(
            "{} created and start running.".format(
                self._gazer_handle_thread.name)
        )

    def process_metrics(self, metrics):
        for task_name, metrics_vector in metrics.items():
            for (time_stamp, metric_dict) in metrics_vector:
                metric_json = json.loads(metric_dict)
                for metric_type, metric_value in \
                        zip(metric_json["MetricType"], metric_json["MetricValue"]):
                    metric = Metric(metric_type, time_stamp, metric_value)
                    for suber in self._subscribers:
                        if any(
                            fnmatch.fnmatch(metric.type, pattern)
                            for pattern in suber.subed_metric_type
                        ):
                            suber.send(metric.type, metric)


class MetricSuber:
    """Metric subscriber."""

    def __init__(self, name, metric_types):
        self._suber_name = name
        if not isinstance(metric_types, (list, tuple)):
            metric_types = [metric_types]
        self._metric_type = set(metric_types)
        self._metric_data = {}
        self._timer = RealTimer()

    def send(self, mtype, metric):
        """dispatch metric to mtype Queue."""
        if not any(fnmatch.fnmatch(mtype, pattern) for pattern in self._metric_type):
            msg = "{} not subscribe {} metric, subscribed metric: {}.".format(
                self._suber_name, mtype, list(self._metric_type)
            )
            logger.warning(msg)
            return False
        if mtype not in self._metric_data:
            self._metric_data[mtype] = queue.Queue()
        self._metric_data[mtype].put(deepcopy(metric))
        return True

    @property
    def name(self):
        """Subscriber name."""
        return self._suber_name

    @property
    def subed_metric_type(self):
        """Subscribed metric type."""
        return self._metric_type

    @property
    def metric_type(self):
        """Real metric type."""
        return self._metric_data.keys()

    def get_metric(self, mtype):
        """Get metric value by metric type."""
        if mtype not in self.metric_type:
            msg = "{} not subscribe {} metric, subscribed metric: {}.".format(
                self._suber_name, mtype, list(self.metric_type)
            )
            logger.debug(msg)
            return None
        mq = self._metric_data[mtype]
        return None if mq.empty() else mq.get()

    def get_snapshot(self): 
        """Get all metrics value by metric type in period of WINDOW_SIZE."""
        result_dict = {}
        for mtype in self.metric_type:
            now = self._timer.now()
            result_list = []
            while not self._metric_data[mtype].empty():
                metric = self._metric_data[mtype].get()
                result_list.append(metric.value)
            result_dict[mtype] = result_list
        return result_dict
