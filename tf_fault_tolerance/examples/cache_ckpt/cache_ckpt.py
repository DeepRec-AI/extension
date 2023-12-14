import datetime
import json
import os
import paiio
import sys
import tensorflow as tf
import tf_fault_tolerance as ft

files = ['./data/sample.csv']

""""
Format of sample.csv
+--------+-------+-----------------------+------------+
| col1   | col2  |        col3           |   col4     |
+--------+-------+-----------------------+------------+
| 0.0    | 0.0   | 0.017179100152531324  |   1.0      |
| 0.0    | 1.0   | 0.823381420409002     |   1.0      |
| 0.0    | 2.0   | 1.6488850495540865    |   1.0      |
+--------+-------+-----------------------+------------+
"""

# Define the input
def input_fn():
    def parse_csv(value):
        defaults = [[0.0] for i in range(0, 4)]
        cols = tf.io.decode_csv(value, record_defaults=defaults)
        return cols[0], cols[1], cols[2], cols[3]

    table_dataset = tf.data.TextLineDataset(files)
    table_dataset = table_dataset.map(parse_csv, num_parallel_calls=28)
    table_dataset = table_dataset.repeat(1000000)
    batch_dataset = table_dataset.batch(128)
    v1, v2, v3, v4 = batch_dataset.make_one_shot_iterator().get_next()
    labels = tf.reshape(tf.cast(v4, tf.int32), [-1])
    features = tf.stack([v1, v2, v3], axis=1)
    return features, labels

# Construct the model
def model_fn(features, labels, global_step):
    W = tf.Variable(tf.zeros([3, 2]))
    b = tf.Variable(tf.zeros([2]))
    pred = tf.matmul(features, W) + b

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=labels))

    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss, global_step=global_step)

    return loss, optimizer

def train(job_name, task_index, cluster, is_chief, target):
    print("start training")

    # Assign io related variables and ops to local worker device
    worker_cpu_device = "/job:%s/task:%d/cpu:%d" % (job_name, task_index, 0)
    with tf.device(worker_cpu_device):
        features, labels = input_fn()

    # Assign global variables to ps nodes
    available_worker_device = "/job:%s/task:%d" % (job_name, task_index)
    print(cluster)
    with tf.device(tf.train.replica_device_setter(worker_device=available_worker_device, cluster=cluster)):
        global_step = tf.Variable(0, name="global_step", trainable=False)
        # Construct the model structure
        loss, optimizer = model_fn(features, labels, global_step)

    scaffold = tf.train.Scaffold()
    hooks=[]
    chief_only_hooks = [ft.train.AsyncCacheCKPTRunnerHook()]
    sess_config = tf.ConfigProto()
    sess_config.device_filters.append("/job:ps")

    local_step = 0
    with tf.train.MonitoredTrainingSession(
            master=target,
            is_chief=is_chief,
            checkpoint_dir="./model_ckpt/",
            save_checkpoint_steps = 100,
            chief_only_hooks=chief_only_hooks,
            hooks=hooks,
            scaffold=scaffold,
            config=sess_config) as mon_sess:
        while True:
            try:
                local_step +=  1
                _, c, g = mon_sess.run([optimizer, loss, global_step])
                if local_step % 10 == 0:
                    print("[%s] step %d, global_step %d, loss is %f" % (datetime.datetime.now().strftime('%b-%d-%y %H:%M:%S'), local_step, g, c))
            except tf.errors.OutOfRangeError:
                print("no more input data to read.")
                break
        print("%d steps finished." % local_step)

def generate_cluster_info(TF_CONFIG):
    tf_config = json.loads(TF_CONFIG)
    cluster_config = tf_config.get('cluster')
    ps_hosts = []
    worker_hosts = []
    chief_hosts = []

    for key, value in cluster_config.items():
        if key == "ps":
            ps_hosts = value
        elif key == "worker":
            worker_hosts = value
        elif key == "chief":
            chief_hosts = value
    if chief_hosts:
        worker_hosts = chief_hosts + worker_hosts

    if not ps_hosts or not worker_hosts:
        print('TF_CONFIG ERROR')
        sys.exit()
    task_config = tf_config.get("task")
    task_type = task_config.get("type")
    task_index = task_config.get("index") + (1 if task_type == "worker" and \
                                             chief_hosts else 0)

    if task_type == "chief":
        task_type = "worker"

    is_chief = True if task_index == 0 else False
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    return task_type, task_index, is_chief, cluster

def main(unused_argv):
    # Get distribute parameter
    TF_CONFIG = os.getenv('TF_CONFIG')
    job_name, task_index, is_chief, cluster_spec = generate_cluster_info(TF_CONFIG)

    # Construct the servers
    server = tf.train.Server(cluster_spec, job_name=job_name, task_index=task_index)

    # Join the ps server
    if job_name == "ps":
        server.join()

    # Start the training
    train(job_name=job_name, task_index=task_index, cluster=cluster_spec, is_chief=is_chief, target=server.target)

if __name__=="__main__":
    tf.app.run()
