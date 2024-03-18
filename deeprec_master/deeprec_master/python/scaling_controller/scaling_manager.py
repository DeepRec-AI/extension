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

import copy
import time

from deeprec_master import pywrap_deeprecmaster
from deeprec_master.python.scaling_controller import StrategyFactory
from deeprec_master.python.metrics_collector import MetricSuber
from deeprec_master.python.ps_service_utilizer import ServiceUtilizer
from deeprec_master.python.utils.logger import logger
from deeprec_master.python.utils import constants
from deeprec_master.python.utils.constants import State, ScalingPlan


class ScalingController:
    def __init__(self, k8s_job_tracker, metrics_collector, jobname, namespace):
        self._job_name = jobname
        self._namespace = namespace
        self._state = State.START
        self._ps_count = None
        self._worker_count = None
        self._scaling_action = ScalingPlan.NORMAL
        self._accumulate_scaling_action = ScalingPlan.NORMAL
        self._last_scaling_time = None
        self._k8s_job_tracker = k8s_job_tracker
        self._metrics_collector = metrics_collector
        self._suber = MetricSuber(name="MetricsPublisher", metric_types=["gazer/*"])
        self._metrics_collector.add_suber(self._suber)
        self._opt_strategy = StrategyFactory.choose_strategy("resource-fit")
        self._ps_service_utilizer = ServiceUtilizer( \
            k8s_job_tracker, self._job_name, self._namespace)

    def try_scaling_ps(self):
        """dynamic embedding server collect metrics"""

        if self._state is State.START:
            if self._wait_resource_for_ps(0):
                tfjob_data = self._k8s_job_tracker.read_job(
                    self._job_name, self._namespace
                )
                ps_num = self._get_tfrole_count(tfjob_data, 'PS')
                worker_count = self._get_tfrole_count(tfjob_data, 'Worker')
                self._ps_service_utilizer.set_init_state(
                    ps_num, worker_count + 1)
                self._state = State.RUNNING
        elif self._state is State.RUNNING:
            if (
                self._last_scaling_time
                and time.time() - self._last_scaling_time < 60 * 5
            ):
                logger.info("ScalingController normal running in 5 minutes.")
                return
            
            job_statistics = self._suber.get_snapshot()
            logger.info("stat_manager statistics: {}".format(job_statistics))
            old_ps_num, ps_limit_kb = self._get_current_ps_num_and_limit()
            new_ps_num, self._scaling_action = self._opt_strategy.make_decision( \
                job_statistics, old_ps_num, ps_limit_kb)
            if self._scaling_action is ScalingPlan.NORMAL:
                logger.info("ScalingController normal running.")
            else:
                tfjob_data = self._k8s_job_tracker.read_job(
                    self._job_name, self._namespace)
                previous_tfjob_data = copy.deepcopy(tfjob_data)
                self._worker_count = self._get_tfrole_count(
                    tfjob_data, 'Worker')
                self._ps_count = new_ps_num
                if self._scaling_action is ScalingPlan.SCALING_DOWN:
                    # Ensure that ps is running with redundent memory
                    if self._accumulate_scaling_action is ScalingPlan.SCALING_DOWN:
                        logger.info("ScalingController update JobPlan.")
                        self._state = State.SCALING
                    self._last_scaling_time = time.time()
                    self._accumulate_scaling_action = ScalingPlan.SCALING_DOWN
                elif self._scaling_action is ScalingPlan.SCALING_UP:
                    logger.info("ScalingController update JobPlan.")
                    tfjob_data["spec"][constants.TF_REPLICA_SPEC]["PS"]["replicas"] = new_ps_num
                    self._ps_service_utilizer.set_tfconfig("notfound")
                    self._k8s_job_tracker.update_job(
                        self._job_name, self._namespace, tfjob_data, previous_tfjob_data
                    )
                    self._state = State.ACQUIRE_RESOURCE
                    self._accumulate_scaling_action = ScalingPlan.SCALING_UP
        elif self._state is State.ACQUIRE_RESOURCE:
            if self._wait_resource_for_ps(self._ps_count - 1):
                self._state = State.SCALING
        elif self._state is State.SCALING:
            logger.info("ScalingController state SCALING.")
            self._ps_service_utilizer.set_scaling_state(
                self._ps_count, self._worker_count + 1
            )
            self._state = State.UPDATE_SERVER
        elif self._state is State.UPDATE_SERVER:
            logger.info("ScalingController state UPDATE_SERVER.")
            if self._ps_service_utilizer.update_server(self._ps_count, self._worker_count + 1):
                self._ps_service_utilizer.set_ready_state(self._ps_count, self._worker_count + 1)
                self._state = State.RELEASE_RESOURCE
                self._last_scaling_time = time.time()
        elif self._state is State.RELEASE_RESOURCE:
            tfjob_data = self._k8s_job_tracker.read_job(self._job_name, self._namespace)
            old_ps_num = self._get_tfrole_count(tfjob_data, 'PS')
            # remove later
            if self._ps_count < old_ps_num:
                previous_tfjob_data = copy.deepcopy(tfjob_data)
                tfjob_data["spec"]["tfReplicaSpecs"]["PS"]["replicas"] = self._ps_count
                self._k8s_job_tracker.use_init_group_version = False
                self._k8s_job_tracker.update_job(
                    self._job_name, self._namespace, tfjob_data, previous_tfjob_data
                )
                self._k8s_job_tracker.use_init_group_version = True
            self._state = State.RUNNING
        else:
            logger.error("ScalingController abnormal state.")
        return

    def set_dynamic_memory(self, has_ev):
        self._opt_strategy.is_dynamic_memory = has_ev

    def _get_tfrole_count(self, tfjob_data, tfrole):
        tf_replica_specs = tfjob_data["spec"][constants.TF_REPLICA_SPEC]
        worker_meta = tf_replica_specs.get(tfrole, None)
        if worker_meta is None:
            worker_count = 0
        else:
            worker_count = worker_meta['replicas']
        return worker_count

    def _wait_resource_for_ps(self, ps_id):
        self._job_data = self._k8s_job_tracker.read_job(self._job_name, self._namespace)
        self.job_json = self._job_data["spec"][constants.TF_REPLICA_SPEC]
        if 'PS' in self.job_json:
            self.job_json[constants.PS_VERTEX_NAME] = self.job_json['PS']
        if 'Master' in self.job_json:
            self.job_json[constants.MASTER_VERTEX_NAME] = self.job_json['Master']
        if 'Worker' in self.job_json:
            self.job_json[constants.WORKER_VERTEX_NAME] = self.job_json['Worker']
        if 'Chief' in self.job_json:
            self.job_json[constants.CHIEF_VERTEX_NAME] = self.job_json['Chief']
        if 'Evaluator' in self.job_json:
            self.job_json[constants.EVALUATOR_VERTEX_NAME] = self.job_json['Evaluator']
        worker_status = self._k8s_job_tracker.get_pods_status(
            self._job_data["metadata"], self.job_json
        )
        id_string = "{}#{}".format("ps", ps_id)
        for id, worker in worker_status.items():
            if id == id_string:
                for e in worker.spec.containers[0].env:
                    if e.name == "TF_CONFIG":
                        self._ps_service_utilizer.set_tfconfig(e.value)
                        return True
        return False

    def _get_current_ps_num_and_limit(self):
        """get ps num through dag_builder"""
        tfjob_data = self._k8s_job_tracker.read_job(
            self._job_name, self._namespace)
        old_ps_num = self._get_tfrole_count(tfjob_data, 'PS')
        ps_limits_memory = tfjob_data["spec"]["tfReplicaSpecs"]["PS"]["template"][
            "spec"
        ]["containers"][0]["resources"]["requests"]["memory"]
        index = ps_limits_memory.find("Gi")
        ps_limits_memory = int(ps_limits_memory[:index]) * constants.GIGABYTE
        return old_ps_num, ps_limits_memory
