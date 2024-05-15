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

""" consts """
from enum import Enum

PS_VERTEX_NAME = "ps"
MASTER_VERTEX_NAME = "master"
WORKER_VERTEX_NAME = "worker"
CHIEF_VERTEX_NAME = "chief"
EVALUATOR_VERTEX_NAME = "evaluator"
DEEPRECMASTER_ADDR_ENV = "DEEPRECMASTER_ADDR"
DEEPRECMASTER_ANNOTATION = "deeprecmaster"

TF_TYPE_NAME = "tf"
TF_PLURAL_NAME = "tfjobs"
TF_KIND_NAME = "TFJob"
TF_DEFAULT_CONTAINER_NAME = "tensorflow"
TF_PORT_NAME = "tfjob-port"
TF_REPLICA_SPEC = "tfReplicaSpecs"
MEGABYTE = 1024 * 1024
GIGABYTE = 1024 * 1024 * 1024
MAX_SCALING_PS_NUM = 100

class ScalingPlan(Enum):
  NORMAL = 0
  SCALING_UP = 1
  SCALING_DOWN = 2
  MIGRATION = 3

class State(Enum):
  START = 0
  RUNNING = 1
  ACQUIRE_RESOURCE = 2
  SCALING = 3
  UPDATE_SERVER = 4
  RELEASE_RESOURCE = 5


class CRDConst:
    """Constant for CustomResourceDefinitions"""

    GROUP = "kubeflow.org"
    VERSION = "v1"
    INIT_GROUP = "training.pai.ai"
    INIT_VERSION = "v1alpha1"
