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

""" k8s interface wrapper."""

import copy
import sys
import threading
import time
import traceback
from typing import List
import json
from jsonpatch import JsonPatch
import urllib3
import kubernetes
from kubernetes.client.rest import ApiException
from kubernetes.client.models.v1_node import V1Node
from kubernetes.client.api import core_v1_api
from kubernetes import watch

from deeprec_master.python.metrics_collector.metrics_collector import Metric
from deeprec_master.python.utils.retry import RetryCaller
from deeprec_master.python.utils.logger import logger
from deeprec_master.python.utils import constants, kubernetes_config
from deeprec_master.python.master_client.utils import gen_owner_references
from deeprec_master.python.master_client import MasterV1Api

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class KubernetesJobTracker:
    """Tracker for k8s"""

    def __init__(self):
        """
        Initialze a KubernetesJobTracker object
        Args:
          api_client: the client of api
        """
        api_client = kubernetes_config.get_k8s_apiclient()
        if api_client is None:
            raise ValueError("api_client must not be None")
        self.api_client = api_client
        self.job_default_container = constants.TF_DEFAULT_CONTAINER_NAME
        self.master_v1api_instance = MasterV1Api(api_client)
        self.core_v1_api_instance = core_v1_api.CoreV1Api(api_client)
        self.watcher = watch.Watch()
        self._use_init_group_version = False
        self._retryable_call = RetryCaller()

    @property
    def use_init_group_version(self):
        return self._use_init_group_version

    @use_init_group_version.setter
    def use_init_group_version(self, value: bool):
        self._use_init_group_version = value

    def read_job(self, jobname, namespace):
        """
        Read job info
        Args:
          jobname: name of the job
          namespace: namespace of the job
          await_creation: whether to wait longer for the creation of the job upon NotFound error of the api call
        """
        return self.master_v1api_instance.read_namespace(
            jobname, namespace, init=self.use_init_group_version
        )

    def update_job(self, jobname, namespace, new_job, old_job=None):
        """
        Read job info
        Args:
          jobname: name of the job
          namespace: namespace of the job
          new_job: object data of the new job
          old_job: object data of the old job
        """
        if old_job:
            # Using json-patch, https://jsonpatch.com/
            job_to_patch = JsonPatch.from_diff(old_job, new_job)
            if job_to_patch:
                job_to_patch = json.loads(job_to_patch.to_string())
                logger.info("patch: {}".format(job_to_patch))
            else:
                logger.info("try patching empty, print out call stack")
                traceback.print_stack(file=sys.stderr)
                return None
        else:
            job_to_patch = new_job

        # if type(job_to_patch) is list:
        #     job_to_patch = job_to_patch[0]
        return self.master_v1api_instance.patch_namespace(
            jobname, namespace, job_to_patch, init=self.use_init_group_version
        )

    def get_pods_status(self, job_meta, job_spec):
        """Return pod status"""
        podstatus = {}
        pod_slices = self.get_pod_slices(job_meta, job_spec)

        missing_pods = []
        for pod_name, pod_desc in pod_slices.items():
            pod_status, missing_pod = self.get_pod_status(pod_desc)
            if pod_status is not None:
                podstatus[pod_name] = pod_status
            if missing_pod is not None:
                missing_pods.append(missing_pod)
        if missing_pods:
            missing_pods_msg = ",".join(map(str, missing_pods))
            logger.info("pod  %s  not exists, waiting...", missing_pods_msg)
        return podstatus

    def get_pod_slices(self, metadata, job_spec):
        """Return pod slices"""
        pod_slices = {}
        if constants.PS_VERTEX_NAME in job_spec:
            ps_replica_spec = job_spec[constants.PS_VERTEX_NAME]["template"]["spec"]
            for i in range(job_spec[constants.PS_VERTEX_NAME]["replicas"]):
                pod_instance_name = constants.PS_VERTEX_NAME + "#" + str(i)
                pod_slices[pod_instance_name] = self.generate_pod_spec(
                    metadata, job_spec, ps_replica_spec, constants.PS_VERTEX_NAME, i
                )
        if constants.MASTER_VERTEX_NAME in job_spec:
            master_replica_spec = job_spec[constants.MASTER_VERTEX_NAME]["template"][
                "spec"
            ]
            for i in range(job_spec[constants.MASTER_VERTEX_NAME]["replicas"]):
                pod_instance_name = constants.MASTER_VERTEX_NAME + "#" + str(i)
                pod_slices[pod_instance_name] = self.generate_pod_spec(
                    metadata,
                    job_spec,
                    master_replica_spec,
                    constants.MASTER_VERTEX_NAME,
                    i,
                )
        if constants.WORKER_VERTEX_NAME in job_spec:
            worker_replica_spec = job_spec[constants.WORKER_VERTEX_NAME]["template"][
                "spec"
            ]
            for i in range(job_spec[constants.WORKER_VERTEX_NAME]["replicas"]):
                pod_instance_name = constants.WORKER_VERTEX_NAME + "#" + str(i)
                pod_slices[pod_instance_name] = self.generate_pod_spec(
                    metadata,
                    job_spec,
                    worker_replica_spec,
                    constants.WORKER_VERTEX_NAME,
                    i,
                )
        if constants.CHIEF_VERTEX_NAME in job_spec:
            chief_replica_spec = job_spec[constants.CHIEF_VERTEX_NAME]["template"][
                "spec"
            ]
            for i in range(job_spec[constants.CHIEF_VERTEX_NAME]["replicas"]):
                pod_instance_name = constants.CHIEF_VERTEX_NAME + "#" + str(i)
                pod_slices[pod_instance_name] = self.generate_pod_spec(
                    metadata,
                    job_spec,
                    chief_replica_spec,
                    constants.CHIEF_VERTEX_NAME,
                    i,
                )
        if constants.EVALUATOR_VERTEX_NAME in job_spec:
            evaluator_replica_spec = job_spec[constants.EVALUATOR_VERTEX_NAME][
                "template"
            ]["spec"]
            for i in range(job_spec[constants.EVALUATOR_VERTEX_NAME]["replicas"]):
                pod_instance_name = constants.EVALUATOR_VERTEX_NAME + \
                    "#" + str(i)
                pod_slices[pod_instance_name] = self.generate_pod_spec(
                    metadata,
                    job_spec,
                    evaluator_replica_spec,
                    constants.EVALUATOR_VERTEX_NAME,
                    i,
                )

        return pod_slices

    def get_pod_status(self, pod_desc):
        """Return pod status"""
        try:
            resp = self._retryable_call(
                self.core_v1_api_instance.read_namespaced_pod,
                name=pod_desc["metadata"]["name"],
                namespace=pod_desc["metadata"]["namespace"],
            )
        except kubernetes.client.rest.ApiException:
            # logger.info("pod  %s  not exists, waiting...", pod_desc['metadata']['name'])
            # resp2 = self.core_v1_api_instance.create_namespaced_pod(
            #  body=pod_desc,
            #  namespace=pod_desc['metadata']['namespace'])
            return None, pod_desc["metadata"]["name"]
        else:
            # logger.info("pod %s  running", pod_desc['metadata']['name'] )
            return resp, None

    def get_pod_log(self, pod_desc, **kwargs):
        """Return pod log"""
        try:
            if "container" not in kwargs and self.job_default_container is not None:
                kwargs["container"] = self.job_default_container
            resp = self._retryable_call(
                self.core_v1_api_instance.read_namespaced_pod_log,
                name=pod_desc["metadata"]["name"],
                namespace=pod_desc["metadata"]["namespace"],
                **kwargs
            )
        except kubernetes.client.rest.ApiException:
            logger.info("pod  %s  not exists, waiting...",
                        pod_desc["metadata"]["name"])
            return None
        except UnicodeDecodeError as e:
            logger.warning("failed to get pod log {}".format(e))
            return None
        else:
            return resp

    def get_pod_log_v2(self, pod_name, pod_namespace, **kwargs):
        """Return pod log"""
        pod_desc = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {"name": pod_name, "namespace": pod_namespace},
        }
        return self.get_pod_log(pod_desc, **kwargs)

    def get_simple_pod_desc(self, podname, namespace):
        """Return simple pod description"""
        try:
            resp = self.core_v1_api_instance.read_namespaced_pod(
                name=podname, namespace=namespace
            )
        except kubernetes.client.rest.ApiException:
            logger.info("pod  %s  not exists, waiting...", podname)
            return None
        else:
            logger.info("pod %s  running", podname)
            return resp

    def generate_pod_spec(
        self, metadata, job_spec, replica_spec, replica_type, index
    ):  # pylint: disable=unused-argument
        """Generate pod spec"""
        pod_name = metadata["name"] + "-" + replica_type + "-" + str(index)
        labels = {}
        labels["group-name"] = constants.CRDConst.GROUP
        labels["job-name"] = metadata["name"]
        labels["replica-index"] = str(index)
        labels["replica-type"] = replica_type

        pod_spec = copy.deepcopy(replica_spec)
        pod_spec["containers"][0]["ports"] = [
            {"containerPort": 2222, "name": constants.TF_PORT_NAME, "protocol": "TCP"}
        ]

        pod_manifest = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": pod_name,
                "namespace": metadata["namespace"],
                "labels": labels,
            },
            "spec": pod_spec,
        }
        return pod_manifest

    def generate_service_spec(
        self, metadata, job_spec, replica_spec, replica_type, index
    ):  # pylint: disable=unused-argument
        """Generate service spec"""
        service_name = metadata["name"] + "-" + replica_type + "-" + str(index)
        labels = {}
        labels["group-name"] = constants.CRDConst.GROUP
        labels["job-name"] = metadata["name"]
        labels["replica-index"] = str(index)
        labels["replica-type"] = replica_type

        service_manifest = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": service_name,
                "namespace": metadata["namespace"],
                "labels": labels,
            },
            "spec": {
                "clusterIP": "None",
                "ports": [
                    {
                        "name": constants.TF_PORT_NAME,
                        "port": 2222,
                        "protocol": "TCP",
                        "targetPort": 2222,
                    }
                ],
                "selector": {
                    "group-name": labels["group-name"],
                    "job-name": labels["job-name"],
                    "replica-index": labels["replica-index"],
                    "replica-type": labels["replica-type"],
                },
                "sessionAffinity": "None",
                "type": "ClusterIP",
            },
        }
        return service_manifest

    def get_service_slices(self, metadata, job_spec):
        """Return service slices"""
        service_slices = []
        if "ps" in job_spec:
            ps_replica_spec = job_spec["ps"]["template"]["spec"]
            for i in range(job_spec["ps"]["replicas"]):
                service_slices.append(
                    self.generate_service_spec(
                        metadata, job_spec, ps_replica_spec, "ps", i
                    )
                )
        if "master" in job_spec:
            master_replica_spec = job_spec["master"]["template"]["spec"]
            for i in range(job_spec["master"]["replicas"]):
                service_slices.append(
                    self.generate_service_spec(
                        metadata, job_spec, master_replica_spec, "master", i
                    )
                )
        if "worker" in job_spec:
            worker_replica_spec = job_spec["worker"]["template"]["spec"]
            for i in range(job_spec["worker"]["replicas"]):
                service_slices.append(
                    self.generate_service_spec(
                        metadata, job_spec, worker_replica_spec, "worker", i
                    )
                )
        return service_slices

    def check_service(self, service_desc):
        """Check service"""
        try:
            _ = self.core_v1_api_instance.read_namespaced_service(
                name=service_desc["metadata"]["name"],
                namespace=service_desc["metadata"]["namespace"],
            )
        except kubernetes.client.rest.ApiException:
            print(
                "service  "
                + service_desc["metadata"]["name"]
                + " not exists, waiting..."
            )
            # resp2 = core_v1_api_instance.create_namespaced_service(
            #  body=service_desc,
            #  namespace=service_desc['metadata']['namespace'])
        else:
            print("service " + service_desc["metadata"]["name"] + ", running")

    def run_and_check_service(self, service_desc):
        """Run an check the service"""
        try:
            _ = self.core_v1_api_instance.read_namespaced_service(
                name=service_desc["metadata"]["name"],
                namespace=service_desc["metadata"]["namespace"],
            )
        except kubernetes.client.rest.ApiException:
            logger.info(
                "service  %s  not exists, waiting...", service_desc["metadata"]["name"]
            )
            _ = self.core_v1_api_instance.create_namespaced_service(
                body=service_desc, namespace=service_desc["metadata"]["namespace"]
            )
        else:
            logger.info("service %s  running",
                        service_desc["metadata"]["name"])

    def get_pod_name(self, pod_status):
        """Return pod name"""
        return pod_status.metadata.name

    def get_pod_namespace(self, pod_status):
        """Return pod namespace"""
        return pod_status.metadata.namespace

    def is_pod_deleted(self, pod_status):
        """Return whether the pod is deleted"""
        try:
            resp = self.core_v1_api_instance.read_namespaced_pod(
                name=pod_status.metadata.name, namespace=pod_status.metadata.namespace
            )
        except kubernetes.client.rest.ApiException as e:
            if e.status == 404:
                return 1
            return 0
        else:
            logger.info("pod %s is running", pod_status.metadata.name)
            if resp.metadata.uid != pod_status.metadata.uid:
                logger.info("pod %s has been replaced",
                            pod_status.metadata.name)
                return 2
            return 0

    def make_sure_pod_is_deleted(self, pod_status, timeout_in_seconds=300):
        """Double confirm that pod is deleted"""
        start_time = time.time()
        while time.time() - start_time <= timeout_in_seconds:
            pod_stauts_code = self.is_pod_deleted(pod_status)
            if pod_stauts_code != 0:
                return pod_stauts_code
            time.sleep(1)
        logger.exception(
            "timeout, can't observing pod deleted in %s seconds",
            str(timeout_in_seconds),
        )

    def delete_pod(self, pod_status, **kargs):
        """Delete a pod"""
        try:
            _ = self.core_v1_api_instance.delete_namespaced_pod(
                name=pod_status.metadata.name,
                namespace=pod_status.metadata.namespace,
                **kargs
            )
        except kubernetes.client.rest.ApiException:
            logger.info(
                "soemthing wrong happen when deleting pod %s", pod_status.metadata.name
            )
        else:
            logger.info("pod %s is deleting", pod_status.metadata.name)

    def create_pod(self, pod_status):
        """Create a pod"""
        try:
            _ = self.core_v1_api_instance.create_namespaced_pod(
                namespace=pod_status.metadata.namespace, body=pod_status
            )
        except kubernetes.client.rest.ApiException:
            logger.info(
                "soemthing wrong happen when creating pod %s", pod_status.metadata.name
            )
        else:
            logger.info("pod %s is creating", pod_status.metadata.name)

    def force_delete_pod(self, pod_status, respawn):
        """Force delete a pod"""
        self.delete_pod(pod_status, grace_period_seconds=0)
        repalce_result_code = self.make_sure_pod_is_deleted(pod_status)
        if respawn and (repalce_result_code == 1):
            logger.info(
                "pod %s/%s needs to be replaced",
                pod_status.metadata.namespace,
                pod_status.metadata.name,
            )
            self.create_pod(pod_status)

    def force_delete_pods(self, pods_status, respawn):
        """Force delete a list of pods"""
        worker_threads = []
        for pod_status in pods_status:
            t = threading.Thread(
                target=self.force_delete_pod,
                args=(
                    pod_status,
                    respawn,
                ),
            )
            t.start()
            worker_threads.append(t)

        for t in worker_threads:
            t.join()

    def list_node(self):
        """list nodes"""
        node_list = None
        try:
            node_list = self.core_v1_api_instance.list_node()
            logger.debug("get node list:%s", node_list)
        except ApiException as e:
            logger.warning(
                "Exception when calling CoreV1Api->list_node: {}".format(e))
        return node_list

    def worker_node_filter(self, node):
        """Worker node filter"""
        if not isinstance(node, V1Node):
            raise TypeError("worker_node_filter accept V1Node")
        return node.metadata.labels.get("node-role.kubernetes.io/master", None) is None

    def get_worker_node_number(self):
        """Return worker node number"""
        node_num = 0
        node_list = self.list_node()
        for node in node_list.items:
            if self.worker_node_filter(node):
                node_num += 1
        logger.info("worker node number is %d", node_num)
        return node_num
