import tensorflow as tf
import json
import os
import dynamic_embedding_server
# Parse distributed training configuration and generate cluster information
def generate_cluster_info(TF_CONFIG):
    #print(TF_CONFIG)
    tf_config = json.loads(TF_CONFIG)
    cluster_config = tf_config.get('cluster')
    ps_hosts = []
    worker_hosts = []
    chief_hosts = []
    for key, value in cluster_config.items():
        if 'ps' == key:
            ps_hosts = value
        elif 'worker' == key:
            worker_hosts = value
        elif 'chief' == key:
            chief_hosts = value
    if chief_hosts:
        worker_hosts = chief_hosts + worker_hosts

    if not ps_hosts or not worker_hosts:
        print('TF_CONFIG ERROR')
        sys.exit()
    task_config = tf_config.get('task')
    task_type = task_config.get('type')
    task_index = task_config.get('index') + (1 if task_type == 'worker'
                                             and chief_hosts else 0)

    if task_type == 'chief':
        task_type = 'worker'

    is_chief = True if task_index == 0 else False
    cluster = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})
    server = tf.distribute.Server(cluster,
                                  job_name=task_type,
                                  task_index=task_index,
                                  protocol='elastic-grpc')
    if task_type == 'ps':
        server.join()
    elif task_type == 'worker':
        tf_config = {
            'ps_hosts': ps_hosts,
            'worker_hosts': worker_hosts,
            'type': task_type,
            'index': task_index,
            'is_chief': is_chief
        }
        tf_device = tf.device(
            tf.train.replica_device_setter(
                worker_device='/job:worker/task:%d' % task_index,
                cluster=cluster))
        return tf_config, server, tf_device
    else:
        print("Task type or index error.")
        sys.exit()

TF_CONFIG = os.getenv('TF_CONFIG')

tf_config, server, tf_device = generate_cluster_info(TF_CONFIG)
print(tf_config)

with tf_device:

    ev_opt = tf.EmbeddingVariableOption(evict_option=None,
                                        filter_option=None)
    num_ps_replicas = len(tf_config["ps_hosts"])
    partitioner = tf.fixed_size_partitioner(num_ps_replicas)
    #partitioner = tf.min_max_variable_partitioner(
    #max_partitions=num_ps_replicas, min_slice_size=16 << 10)

    with tf.device("/cpu:0"), tf.variable_scope('', partitioner=partitioner):
      var_0 = tf.get_embedding_variable("var_0",
                                        embedding_dim=16,
                                        initializer=tf.ones_initializer(tf.float32),
                                        partitioner=partitioner,
                                        ev_option=ev_opt)
        
      var_1 = tf.get_embedding_variable("var_1",
                                        embedding_dim=8,
                                        initializer=tf.ones_initializer(tf.float32),
                                        partitioner=partitioner,
                                        ev_option=ev_opt)
      var_2 = tf.get_variable("var_2",
                                shape=(30, 16),
                                initializer=tf.ones_initializer(tf.float32),
                                partitioner=partitioner,
                                use_resource=False)

      var_3 = tf.get_variable("var_3",
                                shape=(30, 8),
                                initializer=tf.ones_initializer(tf.float32),
                                partitioner=partitioner,
                                use_resource=False)

      ##推荐使用SparseTensor的表示
      indices_0 = tf.SparseTensor(indices=[[0,0],[1,1],[2,2],[3,3],[4,3]],
                                values=tf.cast([1, 2, 3, 4, 5], tf.dtypes.int64),
                                dense_shape=[5, 5])
      indices_1 = tf.SparseTensor(indices=[[0,0],[1,1],[2,2],[3,3],[4,3]],
                                values=tf.cast([1, 2, 3, 4, 5], tf.dtypes.int64),
                                dense_shape=[5, 5])
      ids = tf.convert_to_tensor(indices_1.values)
      
      flat_ids = tf.reshape(ids, [-1]) 
      ids_1 = tf.convert_to_tensor(indices_0.values)
      flat_ids_1 = tf.reshape(ids_1, [-1])
      sp_weight = tf.SparseTensor(indices=[[0,0],[0,1],[1,1],[2,2],[3,3],[4,3],[4,4]],
                                values=tf.cast([1,3,2,3,4,5,3], tf.dtypes.float32),
                                dense_shape=[5, 5])
      sp_weight_1 = tf.SparseTensor(indices=[[0,0],[0,1],[1,1],[2,2],[3,3],[4,3],[4,4]],
                                values=tf.cast([1,3,2,3,4,5,3], tf.dtypes.float32),
                                dense_shape=[5, 5])

      emb_1 = tf.nn.embedding_lookup(var_0, flat_ids)
      emb_2 = tf.nn.embedding_lookup(var_1, flat_ids_1)
      emb_3 = tf.nn.embedding_lookup_sparse(var_2, indices_0, sp_weights=None)
      emb_4 = tf.nn.embedding_lookup_sparse(var_3, indices_1, sp_weights=None)
      global_step = tf.train.get_or_create_global_step()
      fun = tf.multiply(tf.concat([emb_1, emb_2, emb_3, emb_4], axis=-1), 2.0, name='multiply')
      dnn_input = tf.layers.dense(fun,
                                  units=12,
                                  activation=tf.nn.relu)
      dnn_input = tf.layers.batch_normalization(
                        dnn_input, training=True, trainable=True)
      loss = tf.reduce_sum(dnn_input, name='reduce_sum')
      opt = tf.train.AdamOptimizer(0.01)
      
      
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v, global_step=global_step)
      global_step_add = tf.assign_add(global_step, 1)
    hooks = []
    hooks.append(dynamic_embedding_server.python.elastic_training_hooks.ElasticTrainingHook())
    scaffold = tf.train.Scaffold(
        local_init_op=tf.group([tf.local_variables_initializer()]),
        saver=tf.train.Saver(max_to_keep=2, sharded=True))
    sess_config = tf.ConfigProto()
    if tf_config:
        sess_config.device_filters.append("/job:ps")
    count = 0
    a= tf.assign_add(global_step, 1)
    with tf.train.MonitoredTrainingSession(
            master=server.target if server else '',
            is_chief=tf_config['is_chief'] if tf_config else True,
            hooks=hooks,
            scaffold=scaffold,
            checkpoint_dir='./result',
            save_checkpoint_steps=1000,
            summary_dir='./result',
            save_summaries_steps=1000,
            config=sess_config) as sess:
        while not sess.should_stop():
            graph = tf.get_default_graph()
            if count < 3500:
              
              print(sess.run([loss, train_op]))            
              #sess.run([loss, a])

              #print(" ======================= ")
              #if count % 500 == 0:
              #print(sess.run([emb_1]))
              #print("grad is: ", sess.run([graph.get_tensor_by_name("gradients/embedding_lookup_sparse_1/embedding_lookup/Reshape_1_grad/Reshape:0")]))
              #print("grad_1 is: ", sess.run([graph.get_tensor_by_name("Adam/update_var_1/part_0/UnsortedSegmentSum:0")]))
              #print("grad_2 is: ", sess.run([graph.get_tensor_by_name("Adam/update_var_1/part_1/UnsortedSegmentSum:0")]))
              #print("partition_indices is: ", sess.run([graph.get_tensor_by_name("embedding_lookup_sparse/embedding_lookup/Gather:0")]))
                #graph = tf.get_default_graph()
             #print(" ======================= ")
                #print(sess.run([graph.get_tensor_by_name("gradients/embedding_lookup_sparse_1_grad/SparseSegmentMeanGrad:0")]))
                #print(sess.run([graph.get_tensor_by_name("gradients/dense/kernel/ConcatPartitions/concat_grad/ConcatOffset:1")]))
                #print(sess.run([graph.get_tensor_by_name("var_2/part_0/read:0")]))
                #print(sess.run([graph.get_tensor_by_name("Adam/update_var_1/part_1/Unique:0")]))
                #print(sess.run([graph.get_tensor_by_name("Adam/update_var_1/part_1/Unique:1")]))
              count+=1
            else:
              exit(0)


