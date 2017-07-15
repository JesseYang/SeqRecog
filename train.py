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

from tensorflow.contrib.layers import variance_scaling_initializer
from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

from cfgs.config import cfg
from reader import Data, CTCBatchData
from mapper import *

class SkipInputRNNCell(tf.contrib.rnn.core_rnn_cell.BasicRNNCell):
# class SkipInputRNNCell(tf.contrib.rnn.BasicRNNCell):
    def __call__(self, inputs, state, scope=None):
        """RNN skipping input projection: output = new_state = activation(input + U * state)."""
        with tf.variable_scope(scope or type(self).__name__):  # "SkipInputRNNCell"
            weights = tf.get_variable("Mat", [self._num_units, self._num_units], dtype=state.dtype)
            state_proj = tf.matmul(state, weights)
            # output = self._activation(state_proj + inputs)
            output = tf.clip_by_value(state_proj + inputs, 0, 20)
        return output, output

@layer_register()
def BiRnn(x, cell_fw, cell_bw, seqlen, initial_fw=None, initial_bw=None):
    if initial_fw == None:
        initial_fw = cell_fw.zero_state(tf.shape(x)[0], tf.float32)
    if initial_bw == None:
        initial_bw = cell_bw.zero_state(tf.shape(x)[0], tf.float32)
    x, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, x, seqlen, initial_fw, initial_bw, dtype=tf.float32)
    x = tf.add(x[0], x[1], "add")
    return x

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
        return [InputDesc(tf.float32, [None, cfg.input_height, 157, cfg.input_channel], 'feat'),   # bxmaxseqx39
                InputDesc(tf.int64, None, 'labelidx'),  # label is b x maxlen, sparse
                InputDesc(tf.int32, None, 'labelvalue'),
                InputDesc(tf.int64, None, 'labelshape'),
                InputDesc(tf.int32, [None], 'seqlen'),   # b
                ]

    # def _build_graph(self, input_vars):
    def _build_graph(self, inputs):
        image, labelidx, labelvalue, labelshape, seqlen = inputs
        tf.summary.image('input_img', image)
        label = tf.SparseTensor(labelidx, labelvalue, labelshape)
        # l = l / 255.0 * 2 - 1

        self.batch_size = tf.shape(image)[0]

        def shortcut(l, n_in, n_out, stride):
            if n_in != n_out:
                return Conv2D('convshortcut', l, n_out, 1, stride=stride)
            else:
                return l

        def basicblock(l, ch_out, stride, preact):
            ch_in = l.get_shape().as_list()[1]
            if preact == 'both_preact':
                l = BNReLU('preact', l)
                input = l
            elif preact != 'no_preact':
                input = l
                l = BNReLU('preact', l)
            else:
                input = l
            l = Conv2D('conv1', l, ch_out, 3, stride=stride, nl=BNReLU)
            l = Conv2D('conv2', l, ch_out, 3)
            return l + shortcut(input, ch_in, ch_out, stride)

        def bottleneck(l, ch_out, stride, preact):
            ch_in = l.get_shape().as_list()[1]
            if preact == 'both_preact':
                l = BNReLU('preact', l)
                input = l
            elif preact != 'no_preact':
                input = l
                l = BNReLU('preact', l)
            else:
                input = l
            l = Conv2D('conv1', l, ch_out, 1, nl=BNReLU)
            l = Conv2D('conv2', l, ch_out, 3, stride=stride, nl=BNReLU)
            l = Conv2D('conv3', l, ch_out * 4, 1)
            return l + shortcut(input, ch_in, ch_out * 4, stride)

        def layer(l, layername, block_func, features, count, stride, first=False):
            with tf.variable_scope(layername):
                with tf.variable_scope('block0'):
                    l = block_func(l, features, stride,
                                   'no_preact' if first else 'both_preact')
                for i in range(1, count):
                    with tf.variable_scope('block{}'.format(i)):
                        l = block_func(l, features, 1, 'default')
                return l

        net_cfg = {
            18: ([2, 2, 2, 2], basicblock),
            34: ([3, 4, 6, 3], basicblock),
            50: ([3, 4, 6, 3], bottleneck),
            101: ([3, 4, 23, 3], bottleneck)
        }
        defs, block_func = net_cfg[18]

        with argscope(Conv2D, nl=tf.identity, use_bias=False, padding='SAME',
                      W_init=variance_scaling_initializer(mode='FAN_OUT')), \
                argscope([Conv2D, BatchNorm], data_format='NHWC'):
            feature = (LinearWrap(image)
                      .apply(layer, 'group0', block_func, 64, defs[0], 1, first=True)
                      .apply(layer, 'group1', block_func, 128, defs[1], 2)
                      .apply(layer, 'group2', block_func, 256, defs[2], 2)
                      .apply(layer, 'group3', block_func, 512, defs[3], 2)
                      .BNReLU('bnlast')())

        seqlen = feature.get_shape()[2]
        # feature_size = feature.get_shape()[1] * feature.get_shape()[3]
        feature_size = 7 * 512

        # cnn part
        # width_shrink = 0
        # with tf.variable_scope('cnn') as scope:
        #     feature_height = cfg.input_height
        #     for i, kernel_height in enumerate(cfg.cnn.kernel_heights):
        #         out_channel = cfg.cnn.channels[i]
        #         kernel_width = cfg.cnn.kernel_widths[i]
        #         l = Conv2D('conv.{}'.format(i),
        #                    l,
        #                    out_channel,
        #                    (kernel_height, kernel_width),
        #                    cfg.cnn.padding)
        #         if cfg.cnn.with_bn:
        #             l = BatchNorm('bn.{}'.format(i), l)
        #         l = tf.clip_by_value(l, 0, 20, "clipped_relu.{}".format(i))
        #         if cfg.cnn.padding == "VALID":
        #             feature_height = feature_height - kernel_height + 1
        #         width_shrink += kernel_width - 1

        #     feature_size = feature_height * out_channel

        # seqlen = tf.subtract(seqlen, width_shrink)

        # rnn part
        l = tf.transpose(feature, perm=[0, 2, 1, 3])
        l = tf.reshape(l, [self.batch_size, -1, feature_size])
        with tf.variable_scope('rnn') as scope:
            for i in range(cfg.rnn.hidden_layers_no):
                # for each rnn layer with sequence-wise batch normalization
                # 1. do the linear projection
                mat = tf.get_variable("linear.{}".format(i), [feature_size, cfg.rnn.hidden_size], dtype=tf.float32)
                l = tf.reshape(l, [-1, feature_size])
                l = tf.matmul(l, mat)
                # 2. sequence-wise batch normalization
                l = BatchNorm('bn.{}'.format(i), l)
                l = tf.reshape(l, [self.batch_size, -1, cfg.rnn.hidden_size])
                # 3. rnn skipping input
                cell_fw = SkipInputRNNCell(cfg.rnn.hidden_size)
                cell_bw = SkipInputRNNCell(cfg.rnn.hidden_size)
                l = BiRnn('bi_rnn.{}'.format(i), l, cell_fw, cell_bw, seqlen)
                feature_size = cfg.rnn.hidden_size

        # fc part
        l = tf.reshape(l, [-1, feature_size])
        output = BatchNorm('bn', l)
        logits = FullyConnected('fc', output, cfg.label_size, nl=tf.identity,
                                W_init=tf.truncated_normal_initializer(stddev=0.01))
        logits = tf.reshape(logits, (self.batch_size, -1, cfg.label_size))

        import pdb
        pdb.set_trace()

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

def get_data(train_or_test, batch_size):
    isTrain = train_or_test == 'train'
    ds = Data(train_or_test, shuffle=isTrain)
    # if isTrain:
    if False:
        augmentors = [
            imgaug.RandomOrderAug(
                [imgaug.Brightness(30, clip=False),
                 imgaug.Contrast((0.8, 1.2), clip=False),
                 imgaug.Saturation(0.4),
                 # rgb-bgr conversion
                 imgaug.Lighting(0.1,
                                 eigval=[0.2175, 0.0188, 0.0045][::-1],
                                 eigvec=np.array(
                                     [[-0.5675, 0.7192, 0.4009],
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203]],
                                     dtype='float32')[::-1, ::-1]
                                 )]),
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
    ds_train = get_data("train", args.batch_size)
    ds_test = get_data("test", args.batch_size)

    return TrainConfig(
        dataflow=ds_train,
        callbacks=[
            ModelSaver(),
            ScheduledHyperParamSetter('learning_rate',
                                      [(0, 1e-4), (60, 3e-5)]),
            InferenceRunner(ds_test, [RecogResult('prediction')]),
            # StatMonitorParamSetter('learning_rate', 'error',
            #                        lambda x: x * 0.2, 0, 5),
            HumanHyperParamSetter('learning_rate'),
            # PeriodicCallback(
            #     InferenceRunner(ds_test, [ScalarStats('error')]), 1),
        ],
        model=Model(),
        max_epoch=100,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default=0)
    parser.add_argument('--batch_size', help='batch size', default=10)
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    logger.auto_set_dir()

    config = get_config(args)
    if args.load:
        config.session_init = SaverRestore(args.load)
    QueueInputTrainer(config).train()
