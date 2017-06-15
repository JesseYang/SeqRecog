#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train.py
# Author: Jesse Yang <jesse.yang1985@gmail.com>

import tensorflow as tf
import numpy as np
import os
import sys
import argparse
from collections import Counter
import operator
import six
from six.moves import map, range
import json

from tensorpack import *
from tensorpack.tfutils.gradproc import *
from tensorpack.utils.globvars import globalns as param
import tensorpack.tfutils.symbolic_functions as symbf
from ctc_data import CTCBatchData
from mapper import *

BATCH = 10

# IMG_HEIGHT = 30
INPUT_CHANNEL = 1


class RecogResult(Inferencer):
    def __init__(self, names, dict_path):
        if not isinstance(names, list):
            self.names = [names]
        else:
            self.names = names
        self.mapper = Mapper(dict_path)

    def _get_output_tensors(self):
        return self.names

    def _before_inference(self):
        self.results = []

    def _datapoint(self, output):
        for prediction in output[0]:
            line = self.mapper.decode_output(prediction)
            self.results.append(line)

    def _after_inference(self):
        for idx, line in enumerate(self.results):
            print(str(idx) + ": " + line)
        return {}

class Model(ModelDesc):

    def __init__(self, params_path, batch_size=BATCH):
        with open(params_path, 'r') as f:
            self.params = json.load(f)
        self.batch_size = batch_size

    def _get_inputs(self):
        return [InputVar(tf.float32, [None, self.params['input_height'], None, INPUT_CHANNEL], 'feat'),   # bxmaxseqx39
                InputVar(tf.int64, None, 'labelidx'),  # label is b x maxlen, sparse
                InputVar(tf.int32, None, 'labelvalue'),
                InputVar(tf.int64, None, 'labelshape'),
                InputVar(tf.int32, [None], 'seqlen'),   # b
                ]

    # def _build_graph(self, input_vars):
    def _build_graph(self, inputs):
        l, labelidx, labelvalue, labelshape, seqlen = inputs
        label = tf.SparseTensor(labelidx, labelvalue, labelshape)

        # cnn part
        width_shrink = 0
        with tf.variable_scope('cnn') as scope:
            feature_height = self.params['input_height']
            for i, kernel_height in enumerate(self.params["cnn"]["kernel_heights"]):
                out_channel = self.params["cnn"]["channels"][i]
                kernel_width = self.params["cnn"]["kernel_widths"][i]
                l = Conv2D('conv.{}'.format(i),
                           l,
                           out_channel,
                           (kernel_height, kernel_width),
                           self.params["cnn"]["padding"])
                if self.params["cnn"]["with_bn"]:
                    l = BatchNorm('bn.{}'.format(i), l)
                l = tf.clip_by_value(l, 0, 20, "clipped_relu.{}".format(i))
                if self.params["cnn"]["padding"] == "VALID":
                    feature_height = feature_height - kernel_height + 1
                width_shrink += kernel_width - 1

            feature_size = feature_height * out_channel

        seqlen = tf.subtract(seqlen, width_shrink)

        # rnn part
        l = tf.transpose(l, perm=[0, 2, 1, 3])
        l = tf.reshape(l, [self.batch_size, -1, feature_size])
        with tf.variable_scope('rnn') as scope:
            for i in range(self.params["rnn"]["nbOfHiddenLayers"]):
                # for each rnn layer with sequence-wise batch normalization
                # 1. do the linear projection
                mat = tf.get_variable("linear.{}".format(i), [feature_size, self.params['rnn']['hiddenSize']], dtype=tf.float32)
                l = tf.reshape(l, [-1, feature_size])
                l = tf.matmul(l, mat)
                # 2. sequence-wise batch normalization
                l = BatchNorm('bn.{}'.format(i), l)
                l = tf.reshape(l, [self.batch_size, -1, self.params['rnn']['hiddenSize']])
                # 3. rnn skipping input
                cell_fw = SkipInputRNNCell(self.params['rnn']['hiddenSize'])
                cell_bw = SkipInputRNNCell(self.params['rnn']['hiddenSize'])
                l = BiRnn('bi_rnn.{}'.format(i), l, cell_fw, cell_bw, seqlen)
                feature_size = self.params['rnn']['hiddenSize']

        # fc part
        l = tf.reshape(l, [-1, self.params['rnn']['hiddenSize']])
        output = BatchNorm('bn', l)
        logits = FullyConnected('fc', output, self.params['label_size'], nl=tf.identity,
                                W_init=tf.truncated_normal_initializer(stddev=0.01))
        logits = tf.reshape(logits, (self.batch_size, -1, self.params['label_size']))

        # ctc output
        loss = tf.nn.ctc_loss(inputs=logits,
                              labels=label,
                              sequence_length=seqlen,
                              time_major=False)
        self.cost = tf.reduce_mean(loss, name='cost')

        # prediction error
        logits = tf.transpose(logits, [1, 0, 2])

        isTrain = get_current_tower_context().is_training
        predictions = tf.to_int32(tf.nn.ctc_greedy_decoder(inputs=logits,
                                                           sequence_length=seqlen)[0][0])

        dense_pred = tf.sparse_tensor_to_dense(predictions, name="prediction")

        err = tf.edit_distance(predictions, label, normalize=True)
        err.set_shape([None])
        err = tf.reduce_mean(err, name='error')
        summary.add_moving_summary(err, self.cost)

    def get_gradient_processor(self):
        return [GlobalNormClip(400)]

    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', 3e-4, summary=True)
        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)

def get_data(path, isTrain):
    ds = LMDBDataPoint(path)
    ds = MapDataComponent(ds, lambda x: x / 255.0 - 0.5)
    ds = CTCBatchData(ds, BATCH)
    if isTrain:
        ds = PrefetchDataZMQ(ds, 1)
    return ds

def get_config(args):
    ds_train = get_data(args.train, isTrain=True)
    ds_test = get_data(args.test, isTrain=False)

    step_per_epochs = ds_train.size()

    return TrainConfig(
        dataflow=ds_train,
        callbacks=[
            ModelSaver(),
            HyperParamSetterWithFunc('learning_rate',
                                     lambda e, x: x / 1.05 ),
            InferenceRunner(ds_test, [RecogResult('prediction', 'dictionary_text')]),
            # StatMonitorParamSetter('learning_rate', 'error',
            #                        lambda x: x * 0.2, 0, 5),
            # HumanHyperParamSetter('learning_rate'),
            # PeriodicCallback(
            #     InferenceRunner(ds_test, [ScalarStats('error')]), 1),
        ],
        model=Model(args.params),
        step_per_epochs=step_per_epochs,
        max_epoch=70,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default=0)
    parser.add_argument('--load', help='load model')
    parser.add_argument('--params', help='path to the params file', default="params_text.json")
    parser.add_argument('--train', help='path to training lmdb', default="train_db")
    parser.add_argument('--test', help='path to testing lmdb', default="test_db")
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    logger.auto_set_dir()

    config = get_config(args)

    ds_train = get_data(args.train, isTrain=True)
    ds_test = get_data(args.test, isTrain=False)

    config = get_config(ds_train, ds_test, args.params)
    if args.load:
        config.session_init = SaverRestore(args.load)
    QueueInputTrainer(config).train()
