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

import threading
import copy
import time
import json
import os

from deeprec_master.python.utils.logger import logger
from deeprec_master.python.utils import constants, kubernetes_config, network, util_func
from deeprec_master.python.metrics_collector import MetricsCollector
from deeprec_master.python.master_client import KubernetesJobTracker
from deeprec_master.python.scaling_controller import ScalingController
from deeprec_master.python.ps_service_utilizer import ServiceUtilizer
from deeprec_master.python.scheduler import Scheduler


class WorkerStatus:
    """Worker status"""

    def __init__(self):
        """Initialize a WorkerStatus object"""
        self._worker_status = {}
        self._worker_status_local = {}
        self._tfconfig = None
        self._chief_status = None
        self._worker_status_changed = False
        self._lock = threading.Lock()

    def update_worker_status(self, updated_str):
        """Update worker status"""
        update_worker_status = {}
        try:
            update_worker_status = json.loads(updated_str)
        except ValueError as err:
            logger.error(
                "Invalid JSON format string: %s, error: %s", updated_str, err)
            return

        with self._lock:
            for taskname, taskstatus in update_worker_status.items():
                self._worker_status[taskname] = taskstatus
            self._worker_status_changed = True

    def get_worker_status(self, force_update=False):
        """Return worker status"""
        with self._lock:
            if self._worker_status_changed or force_update:
                self._worker_status_local = copy.deepcopy(self._worker_status)
                self._worker_status_changed = False

        return self._worker_status_local

    def update_chief_status(self, chief_status):
        """Update chief status"""
        self._chief_status = chief_status

    def get_k8s_simple_workerstatus(self):
        """Return k8s simple worker status"""
        simple_status = {}
        worker_status = self.get_worker_status(force_update=True)
        for _, pod_status in worker_status.items():
            simple_status[pod_status.metadata.name] = {
                "namespace": pod_status.metadata.namespace,
                "status": pod_status.status.phase,
            }
        return simple_status

    def process_request(self, method):
        """Process request"""
        if method == "GetTFConfig":
            if self._tfconfig is not None:
                return worker_pyrpc_pb2.Response(ret=0, result=self._tfconfig)
            return worker_pyrpc_pb2.Response(ret=1, result="")
        return worker_pyrpc_pb2.Response(ret=2, result="unrecognized method")

    def get_running_ps_worker_cnt(self):
        """Return running ps worker count"""
        worker_status = self.get_worker_status()
        current_worker_status = worker_status
        logger.info("fuxi worker status:%s", str(current_worker_status))
        worker_running_cnt = 0
        ps_running_cnt = 0
        for taskname, taskstatus in current_worker_status.items():
            if taskname.find("worker#") == 0 and taskstatus["TaskState"] == "RUNNING":
                worker_running_cnt += 1
            if taskname.find("ps#") == 0 and taskstatus["TaskState"] == "RUNNING":
                ps_running_cnt += 1
        return ps_running_cnt, worker_running_cnt


class JobMaster:
    """Base class for job controller"""

    def __init__(self, jobname, namespace, args):
        self._jobname = jobname
        self._namespace = namespace
        self._k8s_job_tracker = KubernetesJobTracker()
        self._args = args
        self._job_data = None
        self._orig_job_data = None
        self._worker_status = WorkerStatus()
        self._deeprecmaster_service = None

    @property
    def jobname(self):
        return self._jobname

    @property
    def namespace(self):
        return self._namespace

    def submit_plan(self):
        """Submit the current plan for k8s"""
        raise NotImplementedError("must be implemented in descendants")

    def _fetch_job_from_k8s(self):
        """Get job info from k8s"""
        self._job_data = self._k8s_job_tracker.read_job(
            self._jobname, self._namespace)
        if self._orig_job_data is None:
            self._orig_job_data = copy.deepcopy(self._job_data)

    def _push_job_to_k8s(self):
        """Push the job to k8s"""
        logger.info("final job plan %s", self._job_data)
        self._k8s_job_tracker.update_job(
            self._jobname, self._namespace, self._job_data, self._orig_job_data
        )
        self._orig_job_data = copy.deepcopy(self._job_data)

    def _change_deeprecmaster_status(self, new_status):
        """Change the master status to allow master-operator reconcile"""
        if self._job_data["metadata"].get("annotations", None) is None:
            self._job_data["metadata"]["annotations"] = {}
        self._job_data["metadata"]["annotations"]["deeprecmaster"] = new_status

    def get_k8s_tracker(self):
        """Return the k8s tracker"""
        return self._k8s_job_tracker

    def get_worker_status(self, verbose=False):
        """Return worker status"""
        job_meta, job_json_plan = self.prepare_job_meta_and_plan()
        self._worker_status._worker_status = self._k8s_job_tracker.get_pods_status(
            job_meta, job_json_plan
        )  # pylint: disable=protected-access
        if verbose:
            return self._worker_status.get_worker_status()
        return self._worker_status.get_k8s_simple_workerstatus()

    def prepare_job_meta_and_plan(self):
        job_meta = self._job_data["metadata"]
        job_json_plan = self._job_data["spec"][constants.TF_REPLICA_SPEC]
        if 'PS' in job_json_plan:
            job_json_plan[constants.PS_VERTEX_NAME] = job_json_plan['PS']
        if 'Master' in job_json_plan:
            job_json_plan[constants.MASTER_VERTEX_NAME] = job_json_plan['Master']
        if 'Worker' in job_json_plan:
            job_json_plan[constants.WORKER_VERTEX_NAME] = job_json_plan['Worker']
        if 'Chief' in job_json_plan:
            job_json_plan[constants.CHIEF_VERTEX_NAME] = job_json_plan['Chief']
        if 'Evaluator' in job_json_plan:
            job_json_plan[constants.EVALUATOR_VERTEX_NAME] = job_json_plan['Evaluator']
        return job_meta, job_json_plan

    def _patch_master_addr_env_to_pods(self, master_addr):
        """Patching deeprecmaster host to tf"""
        master_addr_env = os.getenv(constants.DEEPRECMASTER_ADDR_ENV)
        if master_addr_env:
            logger.warning(
                "{} Env already exists, value: {}".format(
                    constants.DEEPRECMASTER_ADDR_ENV, master_addr_env
                )
            )
            return

        master_env = {"name": constants.DEEPRECMASTER_ADDR_ENV,
                        "value": master_addr}
        for replica_type, replica_spec in self._job_data["spec"][
            constants.TF_REPLICA_SPEC
        ].items():
            if replica_type != "AIMaster":
                container_spec = replica_spec["template"]["spec"]["containers"][0]
                if "env" not in container_spec:
                    container_spec["env"] = []
                found = False
                for env_pair in container_spec["env"]:
                    if env_pair["name"] == constants.DEEPRECMASTER_ADDR_ENV:
                        env_pair["value"] = master_addr
                        found = True
                        break
                if not found:
                    replica_spec["template"]["spec"]["containers"][0]["env"].append(
                        master_env
                    )


class TfJobMaster(JobMaster):
    LOOP_PERIOD_SECONDS = 10
    def __init__(self, jobname, namespace, args):
        super().__init__(jobname=jobname, namespace=namespace, args=args)
        self._metrics_collector = MetricsCollector()
        self._scaling_controller = ScalingController(
            self._k8s_job_tracker, self._metrics_collector, jobname, namespace)
        if self._is_ps_resource_analyzer_enabled():
            self._ps_service_utilizer = ServiceUtilizer(
                self._k8s_job_tracker, self._jobname, self._namespace)

    def submit_plan(self):
        """ Submit the job plan """
        logger.info("get job from k8s: %s", str(self._job_data))
        if self._is_ps_resource_analyzer_enabled():
            logger.info("DynamicEmbeddingServer is enabled.")
            has_ev = self._ps_service_utilizer.resource_analyze(self._job_data)
            self._scaling_controller.set_dynamic_memory(has_ev)
            self._fetch_job_from_k8s()
        else:
            self._patch_master_addr_env_to_pods(
                self._deeprecmaster_service.addr)
            self._change_deeprecmaster_status("ready")
            self._push_job_to_k8s()
        self._k8s_job_tracker.use_init_group_version = False
        logger.info("submit plan succeed.")

    def start(self):
        """Start the controller"""
        # Using master-operator, the job object has not yet been created at this moment.
        # In this case, Master should push initial plans to master-operator objects
        # (i.e., those with the API group training.pai.ai).
        self._k8s_job_tracker.use_init_group_version = True
        self._fetch_job_from_k8s()
        ip, port = network.get_ip_and_port()
        self._deeprecmaster_service = Scheduler(ip, port)
        logger.info(
            "deeprecmaster rpc service started: %s", self._deeprecmaster_service.addr
        )

    # not a join for tf
    def join(self):
        """Looping to take actions"""
        logger.info("tf on k8s joining")
        while True:
            if self._is_tfjob_completed():
                logger.info("job competed, deeprecmaster exit...")
                os._exit(0)  # pylint: disable=protected-access
            else:
                time.sleep(TfJobMaster.LOOP_PERIOD_SECONDS)
                if self._is_ps_resource_analyzer_enabled():
                    self._scaling_controller.try_scaling_ps()

    def _is_ps_resource_analyzer_enabled(self):
        """
        Return whether ps resource analyzer is enabled.
        """
        return self._args.enable_dynamic_embedding_server
    
    def _is_tfjob_completed(self):
        """Return whether the current tf job is completed"""
        new_job_data = self._k8s_job_tracker.read_job(
            self._jobname, self._namespace)
        if "status" in new_job_data and "completionTime" in new_job_data["status"]:
            return True
        return False
