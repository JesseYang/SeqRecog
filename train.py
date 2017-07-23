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
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

from cfgs.config import cfg
from reader import Data, CTCBatchData
from mapper import *

class RecogResult(Inferencer):
    def __init__(self, names):
        if not isinstance(names, list):
            self.names = [names]
        else:
            self.names = names
        self.mapper = Mapper()

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

    def __init__(self):
        pass
        # self.batch_size = batch_size

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, cfg.input_height, None, cfg.input_channel], 'feat'),   # bxmaxseqx39
                InputDesc(tf.int64, None, 'labelidx'),  # label is b x maxlen, sparse
                InputDesc(tf.int32, None, 'labelvalue'),
                InputDesc(tf.int64, None, 'labelshape'),
                InputDesc(tf.int32, [None], 'seqlen'),   # b
                ]

    # def _build_graph(self, input_vars):
    def _build_graph(self, inputs):
        l, labelidx, labelvalue, labelshape, seqlen = inputs
        tf.summary.image('input_img', l)
        label = tf.SparseTensor(labelidx, labelvalue, labelshape)
        l = tf.cast(l, tf.float32)
        l = l / 255.0 * 2 - 1

        self.batch_size = tf.shape(l)[0]

        # cnn part
        width_shrink = 0
        with tf.variable_scope('cnn') as scope:
            feature_height = cfg.input_height
            for i, kernel_height in enumerate(cfg.cnn.kernel_heights):
                out_channel = cfg.cnn.channels[i]
                kernel_width = cfg.cnn.kernel_widths[i]
                l = Conv2D('conv.{}'.format(i),
                           l,
                           out_channel,
                           (kernel_height, kernel_width),
                           cfg.cnn.padding)
                if cfg.cnn.with_bn:
                    l = BatchNorm('bn.{}'.format(i), l)
                l = tf.clip_by_value(l, 0, 20, "clipped_relu.{}".format(i))
                if cfg.cnn.padding == "VALID":
                    feature_height = feature_height - kernel_height + 1
                width_shrink += kernel_width - 1

            feature_size = feature_height * out_channel

        seqlen = tf.subtract(seqlen, width_shrink)

        # rnn part
        l = tf.transpose(l, perm=[0, 2, 1, 3])
        l = tf.reshape(l, [self.batch_size, -1, feature_size])

        if cfg.rnn.hidden_layers_no > 0:
            cell_fw = [tf.nn.rnn_cell.BasicLSTMCell(cfg.rnn.hidden_size) for _ in range(cfg.rnn.hidden_layers_no)]
            cell_bw = [tf.nn.rnn_cell.BasicLSTMCell(cfg.rnn.hidden_size) for _ in range(cfg.rnn.hidden_layers_no)]
            l = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cell_fw, cell_bw, l, dtype=tf.float32)
            feature_size = cfg.rnn.hidden_size

        # fc part
        l = tf.reshape(l[0], [-1, 2 * feature_size])
        # l = tf.reshape(l, [-1, feature_size])
        output = BatchNorm('bn', l)
        logits = FullyConnected('fc', output, cfg.label_size, nl=tf.identity,
                                W_init=tf.truncated_normal_initializer(stddev=0.01))
        logits = tf.reshape(logits, (self.batch_size, -1, cfg.label_size))

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
        # predictions = tf.to_int32(tf.nn.ctc_beam_search_decoder(inputs=logits,
        #                                                    sequence_length=seqlen)[0][0])

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

def get_data(train_or_test, batch_size):
    isTrain = train_or_test == 'train'
    ds = Data(train_or_test, shuffle=isTrain)
    if isTrain:
        augmentors = [
            imgaug.ToFloat32(),
            imgaug.RandomOrderAug(
                [imgaug.Brightness(30, clip=False),
                 imgaug.Contrast((0.8, 1.2), clip=False)]),
            imgaug.Clip(),
            imgaug.ToUint8(),
        ]
    else:
        augmentors = []
    ds = AugmentImageComponent(ds, augmentors) 
    ds = CTCBatchData(ds, batch_size)
    if isTrain:
        # ds = PrefetchDataZMQ(ds, min(6, multiprocessing.cpu_count()))
        ds = PrefetchDataZMQ(ds, 1)
    return ds

def get_config(args):
    ds_train = get_data("train", int(args.batch_size))
    ds_test = get_data("test", args.batch_size)

    return TrainConfig(
        dataflow=ds_train,
        callbacks=[
            ModelSaver(),
            ScheduledHyperParamSetter('learning_rate',
                                      [(0, 1e-5), (3, 3e-5), (6, 6e-5), (10, 1e-4), (60, 1e-5)]),
            # HyperParamSetterWithFunc('learning_rate',
            #                          lambda e, x: x / 1.05 ),
            # InferenceRunner(ds_test, [RecogResult('prediction')]),
            # StatMonitorParamSetter('learning_rate', 'error',
            #                        lambda x: x * 0.2, 0, 5),
            HumanHyperParamSetter('learning_rate'),
            # PeriodicCallback(
            #     InferenceRunner(ds_test, [ScalarStats('error')]), 1),
        ],
        model=Model(),
        max_epoch=70,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default=0)
    parser.add_argument('--batch_size', help='batch size', default=64)
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    logger.auto_set_dir()

    config = get_config(args)
    if args.load:
        config.session_init = SaverRestore(args.load)
    QueueInputTrainer(config).train()
