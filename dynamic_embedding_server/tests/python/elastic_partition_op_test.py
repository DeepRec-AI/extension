#!/usr/bin/env python

# Copyright 2023 Alibaba Group Holding Limited. All Rights Reserved.
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

"""Test for elastic partition op."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dynamic_embedding_server.python import elastic_partition_op

import os
import json

from tensorflow.python.platform import test
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import dtypes

class ElasticPartitionOpTest(test.TestCase):
    def testElasticPartion1D(self):
        with self.session(use_gpu=False) as sess:
            data = constant_op.constant([0, 13, 2, 39, 4, 17], dtype=dtypes.int64)
            indices = constant_op.constant([0, 1, 2, 3, 4, 5])
            partitions = elastic_partition_op.elastic_partition(
                data, indices, num_partitions=4)
            partition_vals, partition_ids = self.evaluate(partitions)

            self.assertEqual(4, len(partition_vals))
            self.assertAllEqual([0, 4], partition_vals[0])
            self.assertAllEqual([13, 17], partition_vals[1])
            self.assertAllEqual([2], partition_vals[2])
            self.assertAllEqual([39], partition_vals[3])
            
            dp_indices = constant_op.constant([0, 1, 2, 3, 0, 1])
            partitions = data_flow_ops.dynamic_partition(
                data, dp_indices, num_partitions=4)
            dp_partition_vals = self.evaluate(partitions)
            self.assertEqual(4, len(dp_partition_vals))
            self.assertAllEqual(dp_partition_vals[0], partition_vals[0])
            self.assertAllEqual(dp_partition_vals[1], partition_vals[1])
            self.assertAllEqual(dp_partition_vals[2], partition_vals[2])
            self.assertAllEqual(dp_partition_vals[3], partition_vals[3])

    
    def testElasticPartion2D(self):
        with self.session(use_gpu=False) as sess:
            data = constant_op.constant([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11],
                                        [12, 13, 14], [15, 16, 17]],
                                        dtype=dtypes.int64)
            indices = constant_op.constant([0, 1, 2, 3, 4, 5])
            partitions = elastic_partition_op.elastic_partition(
                data, indices, num_partitions=4)
            partition_vals, partition_ids = self.evaluate(partitions)

            self.assertEqual(4, len(partition_vals))
            self.assertAllEqual([[0, 1, 2], [12, 13, 14]], partition_vals[0])
            self.assertAllEqual([[3, 4, 5], [15, 16, 17]], partition_vals[1])
            self.assertAllEqual([[6, 7, 8]], partition_vals[2])
            self.assertAllEqual([[9, 10, 11]], partition_vals[3])
            
            dp_indices = constant_op.constant([0, 1, 2, 3, 0, 1])
            partitions = data_flow_ops.dynamic_partition(
                data, dp_indices, num_partitions=4)
            dp_partition_vals = self.evaluate(partitions)
            self.assertEqual(4, len(dp_partition_vals))
            self.assertAllEqual(dp_partition_vals[0], partition_vals[0])
            self.assertAllEqual(dp_partition_vals[1], partition_vals[1])
            self.assertAllEqual(dp_partition_vals[2], partition_vals[2])
            self.assertAllEqual(dp_partition_vals[3], partition_vals[3])

if __name__ == "__main__":
    os.environ["TF_CONFIG"] = json.dumps({
        "cluster": {
            "worker": ["localhost:20086", "localhost:20087"],
            "ps": ["localhost:10086", "localhost:10087"]
        },
    "task": {"type": "worker", "index": 0}
    })
    test.main()