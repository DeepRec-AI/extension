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
""" arguments for deeprecmaster """

import argparse

from deeprec_master.python.utils.config import JobMonitorDefaultConfig as JMDefaultConfig

class ArgsConst:
  CKPT_TIME_LIMIT = 30         # 30 s

def str2bool(v):
  if isinstance(v, bool):
    return v
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  if v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
  """ parse command line arguments """
  parser = argparse.ArgumentParser(description="deeprecmaster arguments", allow_abbrev=False)
  parser.add_argument('--enable-chief-failover', type=str2bool, default=JMDefaultConfig.ENABLE_CHIEF_FAILOVER, \
    help='whether enable chief failover.')
  parser.add_argument('--enable-ps-failover', type=str2bool, default=JMDefaultConfig.ENABLE_PS_FAILOVER, \
    help='whether enable ps failover.')
  parser.add_argument('--max-waiting-time-for-chief-ps-failover', type=int, default=JMDefaultConfig.MAX_WAITING_TIME_FOR_CHIEF_PS_FAILOVER, \
    help='max waiting time for chief/ps failover.')
  parser.add_argument('--enable-dynamic-embedding-server', type=str2bool, default=JMDefaultConfig.ENABLE_DYNAMIC_EMBEDDING_SERVER, \
    help='whether enable ps failover.')

  args = parser.parse_args()
  return args
