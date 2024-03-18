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

"""Tests for redistribution ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dynamic_embedding_server.python import redistribution_ops

import os
import json
import numpy as np

from tensorflow.python.platform import test
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import embedding_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import kv_variable_ops
from tensorflow.python.training import training_util
from tensorflow.python.training import adagrad

class RedistributionOpsTest(test.TestCase):
    def testEVRedistributionScalingUp(self):
        origin_partition_num = 2
        partitioner = partitioned_variables.fixed_size_partitioner(origin_partition_num)
        with ops.device("/cpu:0"):
            var_0 = variable_scope.get_embedding_variable("var_0",
                                        embedding_dim=8,
                                        initializer=init_ops.ones_initializer(dtypes.float32),
                                        partitioner=partitioner,
                                        ev_option = variables.EmbeddingVariableOption(filter_option=None))
        partition_num_ph = array_ops.placeholder(dtypes.int32, name='partition_num')
        emb = embedding_ops.embedding_lookup(var_0, math_ops.cast([0,1,2,3,4,5], dtypes.int64))
        fun = math_ops.multiply(emb, 2.0, name='multiply')
        loss = math_ops.reduce_sum(fun, name='reduce_sum')
        gs = training_util.get_or_create_global_step()
        opt = adagrad.AdagradOptimizer(0.1)
        g_v = opt.compute_gradients(loss)
        train_op = opt.apply_gradients(g_v, gs)
        partition_num = 3
        variable_list = var_0._get_variable_list()
        init = variables.global_variables_initializer()
        shape = variable_list[0].get_dynamic_shape()
        shape_1 = variable_list[1].get_dynamic_shape()
        

        imported_keys = [math_ops.cast([0,1,2,3,4,5], dtypes.int64)]
        imported_values = [math_ops.cast([0,1,2,3,4,5], dtypes.float32)]
        imported_versions = [math_ops.cast([0,1,2,3,4,5], dtypes.int64)]
        imported_freqs = [math_ops.cast([0,1,2,3,4,5], dtypes.int64)]
        for idx, embedding_var in enumerate(variable_list):
            unneeded_ids, unneeded_values, unneeded_versions, unneeded_freqs = \
                redistribution_ops.kv_resource_filter(embedding_var, partition_num_ph, dtypes.int64, dtypes.float32, idx, partition_num)
            imported_keys.extend(unneeded_ids)
            imported_values.extend(unneeded_values)
            imported_freqs.extend(unneeded_freqs)
            imported_versions.extend(unneeded_versions)
        
        run_ops_list= []
        for idx, embedding_var in enumerate(variable_list):
            a = redistribution_ops.kv_resource_mul_import( \
                embedding_var, partition_num_ph, imported_keys, imported_values,
                imported_versions, imported_freqs, partition_id=idx)
            run_ops_list.append(a)

        with self.test_session() as sess:
            sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
            sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
            sess.run([init])
            tmp_shape, tmp_shape_1 = sess.run([shape, shape_1])
            self.assertAllEqual(np.array([0,8]), tmp_shape)
            self.assertAllEqual(np.array([0,8]), tmp_shape_1)
            sess.run([emb, train_op, loss])
            tmp_shape, tmp_shape_1 = sess.run([shape, shape_1])
            self.assertAllEqual(np.array([3,8]), tmp_shape)
            self.assertAllEqual(np.array([3,8]), tmp_shape_1)
            sess.run(run_ops_list, feed_dict={partition_num_ph: partition_num})
            tmp_shape, tmp_shape_1 = sess.run([shape, shape_1])
            self.assertAllEqual(np.array([2,8]), tmp_shape)
            self.assertAllEqual(np.array([2,8]), tmp_shape_1)
    
    def testEVRedistributionScalingDown(self):
        origin_partition_num = 3
        partitioner = partitioned_variables.fixed_size_partitioner(origin_partition_num)
        with ops.device("/cpu:0"):
            var_0 = variable_scope.get_embedding_variable("var_0",
                                        embedding_dim=16,
                                        initializer=init_ops.ones_initializer(dtypes.float32),
                                        partitioner=partitioner,
                                        ev_option = variables.EmbeddingVariableOption(filter_option=None))
        partition_num_ph = array_ops.placeholder(dtypes.int32, name='partition_num')
        emb = embedding_ops.embedding_lookup(var_0, math_ops.cast([0,1,2,3,4,5], dtypes.int64))
        fun = math_ops.multiply(emb, 2.0, name='multiply')
        loss = math_ops.reduce_sum(fun, name='reduce_sum')
        gs = training_util.get_or_create_global_step()
        opt = adagrad.AdagradOptimizer(0.1)
        g_v = opt.compute_gradients(loss)
        train_op = opt.apply_gradients(g_v, gs)
        partition_num = 2
        variable_list = var_0._get_variable_list()
        init = variables.global_variables_initializer()
        shape = variable_list[0].get_dynamic_shape()
        shape_1 = variable_list[1].get_dynamic_shape()
        

        imported_keys = []
        imported_values = []
        imported_versions = []
        imported_freqs = []
        for idx, embedding_var in enumerate(variable_list):
            unneeded_ids, unneeded_values, unneeded_versions, unneeded_freqs = \
                redistribution_ops.kv_resource_filter(embedding_var, partition_num_ph, dtypes.int64, dtypes.float32, idx, partition_num)
            imported_keys.extend(unneeded_ids)
            imported_values.extend(unneeded_values)
            imported_freqs.extend(unneeded_freqs)
            imported_versions.extend(unneeded_versions)
        
        run_ops_list= []
        for idx, embedding_var in enumerate(variable_list):
            sorted_imported_keys = [imported_keys[idx]]
            sorted_imported_values = [imported_values[idx]]
            sorted_imported_versions = [imported_versions[idx]]
            sorted_imported_freqs = [imported_freqs[idx]]
            for i in range(origin_partition_num):
                if i != idx:
                    sorted_imported_keys.append(imported_keys[i])
                    sorted_imported_values.append(imported_values[i])
                    sorted_imported_versions.append(imported_versions[i])
                    sorted_imported_freqs.append(imported_freqs[i])

            a = redistribution_ops.kv_resource_mul_import( \
                embedding_var, partition_num_ph, sorted_imported_keys, sorted_imported_values,
                sorted_imported_versions, sorted_imported_freqs, partition_id=idx)
            run_ops_list.append(a)

        with self.test_session() as sess:
            sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
            sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
            sess.run([init])
            tmp_shape, tmp_shape_1 = sess.run([shape, shape_1])
            self.assertAllEqual(np.array([0,16]), tmp_shape)
            self.assertAllEqual(np.array([0,16]), tmp_shape_1)
            sess.run([emb, train_op, loss])
            tmp_shape, tmp_shape_1 = sess.run([shape, shape_1])
            self.assertAllEqual(np.array([2,16]), tmp_shape)
            self.assertAllEqual(np.array([2,16]), tmp_shape_1)
            sess.run(run_ops_list, feed_dict={partition_num_ph: partition_num})
            tmp_shape, tmp_shape_1 = sess.run([shape, shape_1])
            self.assertAllEqual(np.array([3,16]), tmp_shape)
            self.assertAllEqual(np.array([3,16]), tmp_shape_1)

    def testReAssignScalingUp(self):
        origin_partition_num = 2
        partitioner = partitioned_variables.fixed_size_partitioner(origin_partition_num)
        with ops.device("/cpu:0"):
            var_0 = variable_scope.get_variable("var_0",
                                        shape=(12, 8),
                                        initializer=init_ops.ones_initializer(dtypes.float32),
                                        partitioner=partitioner,
                                        use_resource=False)
        emb = embedding_ops.embedding_lookup(var_0, math_ops.cast([0,1,2,5,6,9], dtypes.int64))
        fun = math_ops.multiply(emb, 2.0, name='multiply')
        loss = math_ops.reduce_sum(fun, name='reduce_sum')
        gs = training_util.get_or_create_global_step()
        opt = adagrad.AdagradOptimizer(0.1)
        g_v = opt.compute_gradients(loss)
        train_op = opt.apply_gradients(g_v, gs)

        partition_num_ph = array_ops.placeholder(dtypes.int32, name='partition_num')
        total_value = var_0.as_tensor()
        init = variables.global_variables_initializer()
        partition_num_ph = array_ops.placeholder(dtypes.int32, name='partition_num')
        partition_num = 3
        variable_list = var_0._get_variable_list()
        run_ops_list = []
        for idx, var in enumerate(variable_list):
            a = redistribution_ops.re_assign(var, total_value, partition_num_ph, idx , idx, origin_partition_num)
            run_ops_list.append(a.op)
        
        ones_arr = np.ones((12, 8))
        def prepare_for_assert(tmp_arr):
            for num in [0,1,3,6,8,10]:
                for j in range(8):
                    tmp_arr[num][j] = 0.90122706
            return tmp_arr
        read_value = variable_list[0].read_value()
        with self.test_session() as sess:
            sess.run([init])
            value = sess.run(total_value)
            self.assertAllCloseAccordingToType(ones_arr, value)
            sess.run([emb, train_op, loss])
            value = sess.run(total_value)
            ones_arr = prepare_for_assert(ones_arr)
            self.assertAllCloseAccordingToType(ones_arr, value)
            sess.run(run_ops_list, feed_dict={partition_num_ph: partition_num})
            tmp_value = sess.run([variable_list[0].read_value(), variable_list[1].read_value()])
            self.assertAllCloseAccordingToType(ones_arr[0:4, :], tmp_value[0])
            self.assertAllCloseAccordingToType(ones_arr[4:8, :], tmp_value[1])
    
    def testReAssignScalingDown(self):
        origin_partition_num = 3
        partitioner = partitioned_variables.fixed_size_partitioner(origin_partition_num)
        with ops.device("/cpu:0"):
            var_0 = variable_scope.get_variable("var_0",
                                        shape=(12, 8),
                                        initializer=init_ops.ones_initializer(dtypes.float32),
                                        partitioner=partitioner,
                                        use_resource=False)
        emb = embedding_ops.embedding_lookup(var_0, math_ops.cast([0,1,2,3,4,5], dtypes.int64))
        fun = math_ops.multiply(emb, 2.0, name='multiply')
        loss = math_ops.reduce_sum(fun, name='reduce_sum')
        gs = training_util.get_or_create_global_step()
        opt = adagrad.AdagradOptimizer(0.1)
        g_v = opt.compute_gradients(loss)
        train_op = opt.apply_gradients(g_v, gs)

        partition_num_ph = array_ops.placeholder(dtypes.int32, name='partition_num')
        total_value = var_0.as_tensor()
        init = variables.global_variables_initializer()
        partition_num = 2
        variable_list = var_0._get_variable_list()
        run_ops_list = []
        for idx, var in enumerate(variable_list):
            a = redistribution_ops.re_assign(var, total_value, partition_num_ph, idx , idx, origin_partition_num)
            run_ops_list.append(a.op)
        
        ones_arr = np.ones((12, 8))
        def prepare_for_assert(tmp_arr):
            for num in [0,1,4,5,8,9]:
                for j in range(8):
                    tmp_arr[num][j] = 0.90122706
            return tmp_arr
        read_value = variable_list[0].read_value()
        with self.test_session() as sess:
            sess.run([init])
            value = sess.run(total_value)
            self.assertAllCloseAccordingToType(ones_arr, value)
            sess.run([emb, train_op, loss])
            value = sess.run(total_value)
            ones_arr = prepare_for_assert(ones_arr)
            self.assertAllCloseAccordingToType(ones_arr, value)
            sess.run(run_ops_list, feed_dict={partition_num_ph: partition_num})
            tmp_value = sess.run([variable_list[0].read_value(), variable_list[1].read_value()])
            self.assertAllCloseAccordingToType(ones_arr[0:6, :], tmp_value[0])
            self.assertAllCloseAccordingToType(ones_arr[6:12, :], tmp_value[1])

    def testReAssignResourceScalingDown(self):
        origin_partition_num = 3
        partitioner = partitioned_variables.fixed_size_partitioner(origin_partition_num)
        with ops.device("/cpu:0"):
            var_0 = variable_scope.get_variable("var_0",
                                        shape=(12, 8),
                                        initializer=init_ops.ones_initializer(dtypes.float32),
                                        partitioner=partitioner,
                                        use_resource=True)
        emb = embedding_ops.embedding_lookup(var_0, math_ops.cast([0,1,2,3,4,5], dtypes.int64))
        fun = math_ops.multiply(emb, 2.0, name='multiply')
        loss = math_ops.reduce_sum(fun, name='reduce_sum')
        gs = training_util.get_or_create_global_step()
        opt = adagrad.AdagradOptimizer(0.1)
        g_v = opt.compute_gradients(loss)
        train_op = opt.apply_gradients(g_v, gs)

        partition_num_ph = array_ops.placeholder(dtypes.int32, name='partition_num')
        total_value = var_0.as_tensor()
        init = variables.global_variables_initializer()
        partition_num = 2
        variable_list = var_0._get_variable_list()
        run_ops_list = []
        for idx, var in enumerate(variable_list):
            a = redistribution_ops.re_assign_resource(var, total_value, partition_num_ph, idx , idx, origin_partition_num)
            run_ops_list.append(a)
        ones_arr = np.ones((12, 8))
        def prepare_for_assert(tmp_arr):
            for num in [0,1,4,5,8,9]:
                for j in range(8):
                    tmp_arr[num][j] = 0.90122706
            return tmp_arr
        with self.test_session() as sess:
            sess.run([init])
            value = sess.run(total_value)
            self.assertAllCloseAccordingToType(ones_arr, value)
            sess.run([emb, train_op, loss])
            value = sess.run(total_value)
            ones_arr = prepare_for_assert(ones_arr)
            self.assertAllCloseAccordingToType(ones_arr, value)
            sess.run(run_ops_list, feed_dict={partition_num_ph: partition_num})
            tmp_value = sess.run([variable_list[0].read_value(), variable_list[1].read_value()])
            self.assertAllCloseAccordingToType(ones_arr[0:6, :], tmp_value[0])
            self.assertAllCloseAccordingToType(ones_arr[6:12, :], tmp_value[1])

    def testReAssignResourceScalingUp(self):
        origin_partition_num = 2
        partitioner = partitioned_variables.fixed_size_partitioner(origin_partition_num)
        with ops.device("/cpu:0"):
            var_0 = variable_scope.get_variable("var_0",
                                        shape=(12, 8),
                                        initializer=init_ops.ones_initializer(dtypes.float32),
                                        partitioner=partitioner,
                                        use_resource=True)
        emb = embedding_ops.embedding_lookup(var_0, math_ops.cast([0,1,2,3,4,5], dtypes.int64))
        fun = math_ops.multiply(emb, 2.0, name='multiply')
        loss = math_ops.reduce_sum(fun, name='reduce_sum')
        gs = training_util.get_or_create_global_step()
        opt = adagrad.AdagradOptimizer(0.1)
        g_v = opt.compute_gradients(loss)
        train_op = opt.apply_gradients(g_v, gs)

        partition_num_ph = array_ops.placeholder(dtypes.int32, name='partition_num')
        total_value = var_0.as_tensor()
        init = variables.global_variables_initializer()
        partition_num_ph = array_ops.placeholder(dtypes.int32, name='partition_num')
        partition_num = 3
        variable_list = var_0._get_variable_list()
        run_ops_list = []
        for idx, var in enumerate(variable_list):
            a = redistribution_ops.re_assign_resource(var, total_value, partition_num_ph, idx , idx, origin_partition_num)
            run_ops_list.append(a)
        ones_arr = np.ones((12, 8))
        def prepare_for_assert(tmp_arr):
            for num in [0,1,2,6,7,8]:
                for j in range(8):
                    tmp_arr[num][j] = 0.90122706
            return tmp_arr
        with self.test_session() as sess:
            sess.run([init])
            value = sess.run(total_value)
            self.assertAllCloseAccordingToType(ones_arr, value)
            sess.run([emb, train_op, loss])
            value = sess.run(total_value)
            ones_arr = prepare_for_assert(ones_arr)
            self.assertAllCloseAccordingToType(ones_arr, value)
            sess.run(run_ops_list, feed_dict={partition_num_ph: partition_num})
            tmp_value = sess.run([variable_list[0].read_value(), variable_list[1].read_value()])
            self.assertAllCloseAccordingToType(ones_arr[0:4, :], tmp_value[0])
            self.assertAllCloseAccordingToType(ones_arr[4:8, :], tmp_value[1])

if __name__ == "__main__":
    os.environ["TF_CONFIG"] = json.dumps({
        "cluster": {
            "worker": ["localhost:20086", "localhost:20087"],
            "ps": ["localhost:10086", "localhost:10087"]
        },
    "task": {"type": "worker", "index": 0}
    })
    test.main()