
# !/usr/bin/env python
# coding=utf-8

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
'''
structure
LR+MLP=
'''
import argparse
import shutil
import sys
import os
import glob
import json
import random
from datetime import date, timedelta

import tensorflow as tf
import gazer

#################### CMD Arguments ####################
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("ps_hosts", '', "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", '', "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", '', "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("num_threads", 10, "Number of threads")
tf.app.flags.DEFINE_integer("embedding_size", 32, "Embedding size")
tf.app.flags.DEFINE_integer("num_epochs", 1, "Number of epochs")
tf.app.flags.DEFINE_integer("batch_size", 128, "batch size")
tf.app.flags.DEFINE_string("deep_layers", '256,128,64', "deep layers")
tf.app.flags.DEFINE_integer("log_steps", 1000, "save summary every steps")
tf.app.flags.DEFINE_integer("throttle_secs", 600, "evaluate every 10mins")
tf.app.flags.DEFINE_string("data_dir", '', "data dir")
tf.app.flags.DEFINE_string("dt_dir", '', "data dt partition")
tf.app.flags.DEFINE_string("model_dir", '', "model check point dir")
tf.app.flags.DEFINE_string("servable_model_dir", '', "export servable model for TensorFlow Serving")
tf.app.flags.DEFINE_string("task_type", 'train', "task type {train, predict, export}")
tf.app.flags.DEFINE_string("model_type", 'wide_n_deep', "model type {'wide', 'deep', 'wide_n_deep'}")
tf.app.flags.DEFINE_boolean("clear_existing_model", False, "clear existing model or not")

###############################################################################
#
#       { < u, a, c, xgb >, y }
#
################################################################################
# There are 13 integer features and 26 categorical features
C_COLUMNS = ['I' + str(i) for i in range(1, 14)]
D_COLUMNS = ['C' + str(i) for i in range(14, 40)]
LABEL_COLUMN = "is_click"
CSV_COLUMNS = [LABEL_COLUMN] + C_COLUMNS + D_COLUMNS
# Columns Defaults
CSV_COLUMN_DEFAULTS = [[0.0]]
C_COLUMN_DEFAULTS = [[0.0] for i in range(13)]
D_COLUMN_DEFAULTS = [[""] for i in range(26)]
CSV_COLUMN_DEFAULTS = CSV_COLUMN_DEFAULTS + C_COLUMN_DEFAULTS + D_COLUMN_DEFAULTS
print(CSV_COLUMN_DEFAULTS)


def input_fn(filenames, num_epochs, batch_size=1):
    def parse_csv(line):
        print('Parsing', filenames)
        columns = tf.decode_csv(line, record_defaults=CSV_COLUMN_DEFAULTS)
        features = dict(zip(CSV_COLUMNS, columns))
        labels = features.pop(LABEL_COLUMN)
        return features, labels

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(filenames)  # can pass one filename or filename list

    # multi-thread pre-process then prefetch
    dataset = dataset.map(parse_csv, num_parallel_calls=10).prefetch(500000)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()

    return features, labels


def build_feature():
    # 1 { continuous base columns }
    deep_cbc = [tf.feature_column.numeric_column(colname) for colname in C_COLUMNS]

    # 2 { categorical base columns }
    deep_dbc = [tf.feature_column.categorical_column_with_hash_bucket(colname, hash_bucket_size=1000*100, dtype=tf.string) for
                colname in D_COLUMNS]

    # 3 { embedding columns }
    deep_emb = [tf.feature_column.embedding_column(c, dimension=FLAGS.embedding_size) for c in deep_dbc]

    # 3 { wide columns and deep columns }
    wide_columns = deep_cbc + deep_dbc
    deep_columns = deep_cbc + deep_emb

    return wide_columns, deep_columns


################################################################################
#
#       f(x) / loss / Optimizer
#
################################################################################
def build_estimator(model_dir, model_type, wide_columns, deep_columns):
    """Build an estimator."""

    if FLAGS.clear_existing_model:
        try:
            shutil.rmtree(model_dir)
        except Exception as e:
            print(e, "at clear_existing_model")
        else:
            print("existing model cleaned at %s" % model_dir)

    hidden_units = map(int, FLAGS.deep_layers.split(","))
    config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'GPU': 0, 'CPU': FLAGS.num_threads}),
        save_checkpoints_steps=FLAGS.log_steps, log_step_count_steps=100,
        save_summary_steps=100)
    # bulid model
    estimator = tf.estimator.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
        config=config)

    return estimator


def main(_):
    # ------check Arguments------
    if FLAGS.dt_dir == "":
        FLAGS.dt_dir = (date.today() + timedelta(-1)).strftime('%Y%m%d')

    print('task_type ', FLAGS.task_type)
    print('model_type ', FLAGS.model_type)
    print('model_dir ', FLAGS.model_dir)
    print('servable_model_dir ', FLAGS.servable_model_dir)
    print('dt_dir ', FLAGS.dt_dir)
    print('data_dir ', FLAGS.data_dir)
    print('num_epochs ', FLAGS.num_epochs)
    print('embedding_size ', FLAGS.embedding_size)
    print('deep_layers ', FLAGS.deep_layers)
    print('batch_size ', FLAGS.batch_size)

    # ------init Envs------
    tr_files = glob.glob("%s/tr*csv" % FLAGS.data_dir)
    random.shuffle(tr_files)
    # for test in PAI with data in OSS
    tr_files = FLAGS.data_dir + "/train.csv"
    print("tr_files:", tr_files)
    va_files = glob.glob("%s/va*csv" % FLAGS.data_dir)
    # for test in PAI with data in OSS
    va_files = FLAGS.data_dir + "/eval.csv"
    print("va_files:", va_files)
    te_files = glob.glob("%s/te*csv" % FLAGS.data_dir)
    print("te_files:", te_files)

    # ------build Tasks------
    # build wide_columns, deep_columns
    wide_columns, deep_columns = build_feature()

    # build model
    wide_n_deep = build_estimator(FLAGS.model_dir, FLAGS.model_type, wide_columns, deep_columns)

    if FLAGS.task_type == "train":
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: input_fn(tr_files, num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size))
        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: input_fn(va_files, num_epochs=1, batch_size=FLAGS.batch_size),
            # exporters = lastest_exporter,
            steps=None,  # evaluate the whole eval file
            start_delay_secs=1800,
            throttle_secs=FLAGS.throttle_secs)  # evaluate every 10min for wide
        tf.estimator.train_and_evaluate(wide_n_deep, train_spec, eval_spec)
    elif FLAGS.task_type == "predict":
        pred = wide_n_deep.predict(input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=FLAGS.batch_size),
                                   predict_keys="probabilities")
        with open(FLAGS.data_dir + "/pred.txt", "w") as fo:
            for prob in pred:
                fo.write("%f\n" % (prob['probabilities'][1]))
    elif FLAGS.task_type == "export_model":
        if FLAGS.model_type == "wide":
            feature_columns = wide_columns
        elif FLAGS.model_type == "deep":
            feature_columns = deep_columns
        elif FLAGS.model_type == "wide_n_deep":
            feature_columns = wide_columns + deep_columns
        feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
        serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
        wide_n_deep.export_savedmodel(FLAGS.servable_model_dir, serving_input_receiver_fn)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
