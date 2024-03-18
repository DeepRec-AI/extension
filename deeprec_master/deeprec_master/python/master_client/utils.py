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

""" Utils for k8s clients. """

import time
import os

from kubernetes.client.models.v1_affinity import V1Affinity
from kubernetes.client.models.v1_node_affinity import V1NodeAffinity
from kubernetes.client.models.v1_node_selector import V1NodeSelector
from kubernetes.client.models.v1_node_selector_requirement import (
    V1NodeSelectorRequirement,
)
from kubernetes.client.models.v1_node_selector_term import V1NodeSelectorTerm
from kubernetes.client.models.v1_owner_reference import V1OwnerReference
from kubernetes.client.rest import ApiException

from deeprec_master.python.utils.logger import logger
from deeprec_master.python.utils.retry import RetryCaller


def gen_owner_references(name, group, version, kind, uid):
    """Generate owner references"""
    api_version = group + "/" + version
    owner_reference = V1OwnerReference(
        api_version=api_version, kind=kind, name=name, uid=uid
    )
    return [owner_reference]


def gen_affinity(node_names):
    """Generate node affinity"""
    if not node_names:
        return None
    nodes_affinity = V1Affinity(
        node_affinity=V1NodeAffinity(
            required_during_scheduling_ignored_during_execution=V1NodeSelector(
                node_selector_terms=[
                    V1NodeSelectorTerm(
                        match_expressions=[
                            V1NodeSelectorRequirement(
                                key="kubernetes.io/hostname",
                                operator="In",
                                values=node_names,
                            )
                        ]
                    )
                ]
            )
        )
    )
    return nodes_affinity


def retryable_k8s_http_call(func, kwargs, await_creation=True):
    """retry multiple times of k8s call fail"""
    count_for_creation = 0
    max_num_for_creation = 50
    wait_creation_time = 1
    call_exception = ""
    retryable_call = RetryCaller(max_retry=4, base_delay=1, max_delay=10)

    while True:
        try:
            return retryable_call(func, **kwargs)
        except ApiException as e:
            call_exception = e
            if await_creation and e.status == 404:
                count_for_creation += 1
                if count_for_creation < max_num_for_creation:
                    time.sleep(wait_creation_time)
                    continue
            break
        except Exception as e:  # pylint: disable=broad-except
            call_exception = e
            break

    logger.error("Failed to call {}, error {}".format(
        func.__name__, call_exception))
    os._exit(-1)  # pylint: disable=protected-access
