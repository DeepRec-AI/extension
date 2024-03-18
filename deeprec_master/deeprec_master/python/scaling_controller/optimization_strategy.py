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

from abc import ABCMeta, abstractmethod
import math
import numpy as np

from deeprec_master.python.utils.logger import logger
from deeprec_master.python.utils import constants


class StrategyBase(metaclass=ABCMeta):
    """Base class of strategy."""
    GAZER_PS_JOB = 'job_ps'
    MEM_TYPE = "mem"
    def __init__(self):
        """"""
        self._is_dynamic_memory = False
        self._stat = {}
        self._delta_stat = {}

    @abstractmethod
    def make_decision(self, statistics, ps_limits_memory, old_ps_num):
        """optimization logic should be implemented in children."""
        logger.fatal("Method not implemented!")
        raise NotImplementedError('Method not implemented!')

    @property
    def is_dynamic_memory(self):
        return self._is_dynamic_memory

    @is_dynamic_memory.setter
    def is_dynamic_memory(self, is_dynamic):
        self._is_dynamic_memory = is_dynamic

    def _parse(self, statistics):
        """Parse gazer statistics."""
        for k, v in statistics.items():
            item = k.split("/")
            # ['gazer', 'cpu/mem/res_mem/graph_duration',
            #  'job_ps/job_chief', 'replica_0',
            #  'task_0/1/2/...', 'device_CPU_0']
            if len(item) != 6:
                logger.warning("parse job_statistics failed.")
                continue
            _ = item[0]           # gazer
            metric_type = item[1]  # cpu/mem/res_mem/graph_duration
            job_name = item[2]    # job_ps/job_chief
            _ = item[3]           # replica_0
            task_id = item[4]     # task_0/1/2/...
            _ = item[5]           # device_CPU_0
            if job_name.find(StrategyBase.GAZER_PS_JOB) == -1:
                continue
            if not job_name in self._stat:
                self._stat[job_name] = {}
            if not task_id in self._stat[job_name]:
                self._stat[job_name][task_id] = {}
            if not metric_type in self._stat[job_name][task_id]:
                self._stat[job_name][task_id][metric_type] = 0
            if metric_type == StrategyBase.MEM_TYPE:
                v = np.array([float(mem) * constants.MEGABYTE for mem in v])
            self._stat[job_name][task_id][metric_type] = v


class ResourceFitStrategy(StrategyBase):
    """ResourceFit Strategy."""

    def __init__(self):
        self._already_scaled_ps = set()
        super().__init__()

    def decide_graph_with_ev(self, ps_limits_memory):
        ps_to_scale_up, ps_to_scale_down, ps_might_oom = set(), set(), set()
        for ps_id, stat in self._stat[StrategyBase.GAZER_PS_JOB].items():
            ps_memory_bytes = stat[StrategyBase.MEM_TYPE]
            if len(ps_memory_bytes) == 0:
                continue
            increase_speed = np.diff(ps_memory_bytes)
            logger.info("current mem is: {} , increase speed is: {}".format(ps_memory_bytes, increase_speed))
            if (ps_memory_bytes[-1] > ps_limits_memory * .8) and (ps_id not in self._already_scaled_ps):
                ps_might_oom.add(ps_id)
            elif (ps_memory_bytes[-1] > ps_limits_memory * .6) and (ps_id not in self._already_scaled_ps):
                ps_to_scale_up.add(ps_id)
            elif ps_memory_bytes[-1] < ps_limits_memory * .4:
                ps_to_scale_down.add(ps_id)
        return ps_to_scale_up, ps_to_scale_down, ps_might_oom

    def make_decision(self, statistics, old_ps_num, ps_limits_memory):
        """optimization logic."""
        self._parse(statistics)
        scaling_action = constants.ScalingPlan.NORMAL

        if 'job_ps' not in self._stat:
            logger.warning(
                "[ResourceFitStrategy] PS role not in collected metrics")
            self._stat = {}
            return old_ps_num, scaling_action

        ####################################
        # There is roughly 3 cases to deal with
        # 1.Imbalanced Resource distribution(some ps have high mem usage while others have low)
        # 2. Excessively high configuration for Parameter Server memory.
        # 3. PS with risk of OOM.
        ####################################
        # total_ps_memory_bytes = ps_limits_memory * old_ps_num
        # total_ps_memory_stats = []

        new_ps_num = old_ps_num
        ps_to_scale_up, ps_to_scale_down, ps_might_oom = self.decide_graph_with_ev(ps_limits_memory)

        if len(ps_might_oom) > 0:
            new_ps_num = old_ps_num + 2
            scaling_action = constants.ScalingPlan.SCALING_UP
            for ps_id in ps_might_oom:
                self._already_scaled_ps.add(ps_id)
        elif len(ps_to_scale_up) >= old_ps_num / 2:
            if old_ps_num <= 4:
                new_ps_num = old_ps_num * 2
            else:
                new_ps_num = math.ceil(old_ps_num * 1.2)
            new_ps_num = min(new_ps_num, constants.MAX_SCALING_PS_NUM)
            scaling_action = constants.ScalingPlan.SCALING_UP
            for ps_id in ps_to_scale_up:
                self._already_scaled_ps.add(ps_id)
        elif len(ps_to_scale_down) > old_ps_num / 2:
            if old_ps_num >= 4:
                new_ps_num = math.ceil(old_ps_num / 2)
            else:
                new_ps_num = 2
            scaling_action = constants.ScalingPlan.SCALING_DOWN

        logger.info(
            "[ResourceFitStrategy] decide new ps_num: {} - Scaling action is: {} ".format(new_ps_num, scaling_action))
        self._stat = {}
        return new_ps_num, scaling_action


class StrategyFactory():
    """Initialization factory of Strategy"""
    @staticmethod
    def choose_strategy(name):
        """Currently only resource-fit is implemented."""
        # pylint: disable=no-else-return
        if name == "resource-fit":
            return ResourceFitStrategy()
        elif name == "performance-fit":
            # TODO(JUNQI): gready search for variable num
            return None
        else:
            raise ValueError('Currently, only `resource-fit` is supported!')
