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

"""redistribution ops"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dynamic_embedding_server import dynamic_embedding_server_ops

def re_assign_resource(variable, value, partition_num_in, partition_id, device_id, prev_part_num):
  return dynamic_embedding_server_ops.re_assign_resource(variable.handle,
                                                         value,
                                                         partition_num_in,
                                                         partition_id=partition_id,
                                                         device_id=device_id,
                                                         partition_nums=prev_part_num)

def re_assign(variable, value, partition_num_in, partition_id, device_id, prev_part_num):
    return dynamic_embedding_server_ops.re_assign(variable,
                                                  value,
                                                  partition_num_in,
                                                  partition_id=partition_id,
                                                  device_id=device_id,
                                                  partition_nums=prev_part_num)

def kv_resource_filter(embedding_var, partition_num_in, key_type, dtype, partition_id, partition_num, device_id):
    return dynamic_embedding_server_ops.kv_resource_filter(embedding_var.handle,
                                                            new_partition_nums=partition_num_in,
                                                            partition_nums=partition_num,
                                                            Tkeys=key_type,
                                                            dtype=dtype,
                                                            partition_id=partition_id,
                                                            device_id=device_id)

def kv_resource_mul_import(embedding_var, partition_num_in, keys, values, versions, freqs, partition_id, device_id, partition_nums):
    return dynamic_embedding_server_ops.kv_resource_mul_import(embedding_var.handle,
                                                               partition_num_in,
                                                                keys,
                                                                values,
                                                                versions,
                                                                freqs,
                                                                partition_id=partition_id,
                                                                device_id=device_id)
    