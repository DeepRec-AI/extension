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
import math
import json
import time
import grpc

from deeprec_master import pywrap_deeprecmaster
from deeprec_master.python.utils.logger import logger
from deeprec_master.python.utils import constants


class ServiceUtilizer:
    def __init__(self, k8s_job_tracker, job_name=None, namespace=None):
        self._ps_resource_analyzer = pywrap_deeprecmaster.get_ps_resource_analyzer()
        self._k8s_job_tracker = k8s_job_tracker
        self._job_name = job_name
        self._namespace = namespace

    def resource_analyze(self, tfjob_data):
        """ parameter server resource analysis for k8s 
            This process contains 
            1. rewrite aimaster status
            2. resource analyse
            3. submit new rewriting tfjobs"""
        previous_tfjob_data = copy.deepcopy(tfjob_data)
        ps_num = tfjob_data['spec'][constants.TF_REPLICA_SPEC]['PS']['replicas']
        if tfjob_data['spec'][constants.TF_REPLICA_SPEC].get('Chief') is not None:
            tfjob_data['spec'][constants.TF_REPLICA_SPEC]['PS']['replicas'] = 0
        else:
            raise RuntimeError("Chief node in ClusterSpec TF Job.")
        tfjob_data['metadata']['annotations'][constants.DEEPRECMASTER_ANNOTATION] = 'ready'
        api_response = self._k8s_job_tracker.update_job(self._job_name, self._namespace,
                                        tfjob_data, previous_tfjob_data)
        self._k8s_job_tracker.use_init_group_version = False
        tfjob_data = self._k8s_job_tracker.read_job(
            self._job_name, self._namespace)
        previous_tfjob_data = copy.deepcopy(tfjob_data)

        psres = self._ps_resource_analyzer.get_estimated_ps_resource()
        has_ev = False
        while not psres.is_initialized():
            logger.info("Resource estimating ...")
            time.sleep(10)
            psres = self._ps_resource_analyzer.get_estimated_ps_resource()

        logger.info("ps analyze num: %d, memory:%d B",
                    psres.ps_num(), psres.ps_memory())
        if psres.ps_memory() > 0:
            ps_limits_memory = self._acquire_ps_limit_mem(tfjob_data)
            ps_num = int(math.ceil(psres.ps_memory() / 0.6 / ps_limits_memory))
            has_ev = psres.has_ev()
            logger.info("ps limits memory: %d B, evaluate ps num: %d.",
                        ps_limits_memory, ps_num)
        tfjob_data['spec']['tfReplicaSpecs']['PS']['replicas'] = ps_num
        
        self._k8s_job_tracker.update_job(self._job_name, self._namespace,
                                        tfjob_data, previous_tfjob_data)
        return has_ev

    def _acquire_ps_limit_mem(self, tfjob_data):
        ps_limit_str = tfjob_data['spec']['tfReplicaSpecs']["PS"]["template"]\
                ["spec"]["containers"][0]["resources"]["requests"]["memory"]
        index = ps_limit_str.find("Gi")
        if index != -1:
            ps_limit = int(ps_limit_str[:index]) * constants.GIGABYTE
        else:
            mi_index = ps_limit_str.find("Mi")
            if mi_index != -1:
                ps_limit = int(ps_limit_str[:mi_index]) * constants.MEGABYTE
            else:
                logger.warning("UNRECOGNIZED ps_limit_str: {}".format(ps_limit_str))
                return -1
        return ps_limit

    def get_state(self):
        return self._ps_resource_analyzer.get_state()

    def set_init_state(self, ps_num, worker_num):
        self._ps_resource_analyzer.set_state(
            pywrap_deeprecmaster.ElasticTrainingState.INIT,
            ps_num, worker_num)

    def set_scaling_state(self, ps_num, worker_num):
        self._ps_resource_analyzer.set_state(
            pywrap_deeprecmaster.ElasticTrainingState.SCALING,
            ps_num, worker_num)

    def set_ready_state(self, ps_num, worker_num):
        self._ps_resource_analyzer.set_state(
            pywrap_deeprecmaster.ElasticTrainingState.READY,
            ps_num, worker_num)

    def get_tfconfig(self):
        return self._ps_resource_analyzer.get_tfconfig()

    def set_tfconfig(self, tf_config):
        self._ps_resource_analyzer.set_tfconfig(tf_config)

    def update_server(self, ps_count, worker_count):
        s = self._ps_resource_analyzer.get_state()
        tfconfig = self._ps_resource_analyzer.get_tfconfig()
        tfconfig = json.loads(tfconfig)
        cluster = tfconfig.get("cluster", {})
        ps_template = cluster["ps"][0]
        ind = ps_template.find("ps")
        new_ps_list = []
        for index in range(ps_count):
            new_ps_list.append(ps_template[:ind] + "ps-" + str(index) + ps_template[ind+4:])
        cluster["ps"] = new_ps_list
        if cluster and s == pywrap_deeprecmaster.ElasticTrainingState.All_SESSION_CLOSED:
            addrs = sum(cluster.values(), [])

            from deeprec_master.python.scaling_controller import elastic_training_pb2
            from deeprec_master.python.scaling_controller import elastic_training_pb2_grpc
            for addr in addrs:
                logger.info("DES calling {}.".format(addr))
                channel = grpc.insecure_channel(addr)
                stub = elastic_training_pb2_grpc.ElasticTrainingServiceStub(channel)
                req = elastic_training_pb2.UpdateServerDefRequest()
                req.cluster_def = json.dumps({"cluster": cluster})
                while True:
                    try:
                        resp = stub.UpdateServerDef(req)
                        break
                    except grpc.RpcError as gre:
                        status_code = gre.code()
                        if status_code in (
                            grpc.StatusCode.DEADLINE_EXCEEDED,
                            grpc.StatusCode.UNAVAILABLE,
                        ):
                            logger.warning("Elastic Server {} is unavailable.".format(addr))
                            time.sleep(10)
                            continue
                        raise gre
            return True
        return False
