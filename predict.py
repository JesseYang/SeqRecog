#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train.py
# Author: Jesse Yang <jesse.yang1985@gmail.com>

import os
import numpy as np
from scipy import misc
import argparse

from tensorpack import *

from train import Model
from mapper import *

def sequence_error_stat(target, prediction):
    d = np.zeros([len(target) + 1, len(prediction) + 1])
    for i in range(len(target) + 1):
        for j in range(len(prediction) + 1):
            if i == 0:
                d[0, j] = j
            elif j == 0:
                d[i, 0] = i

    for i in range(1, len(target) + 1):
        for j in range(1, len(prediction) + 1):
            if target[i - 1] == prediction[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                substitution = d[i - 1, j - 1] + 1
                insertion = d[i, j - 1] + 1
                deletion = d[i - 1, j] + 1
                d[i, j] = np.min([substitution, insertion, deletion])

    return (d[len(target), len(prediction)], len(target))

def predict_one(dict_path, img_path, predict_func, idx):
    img = misc.imread(img_path)
    seqlen = img.shape[1]
    img = np.expand_dims(np.expand_dims(img, axis=2), axis=0)
    img = img / 255.0 - 0.5

    predictions = predict_func([img, [seqlen]])[0]

    mapper = Mapper(dict_path)
    result = mapper.decode_output(predictions[0])
    if idx == None:
        logger.info(img_path)
        logger.info(result)
    else:
        logger.info(str(idx) + ": " + img_path)
        logger.info(str(idx) + ": " + result)
    return result

def predict(args):
    sess_init = SaverRestore(args.model_path)
    model = Model()
    predict_config = PredictConfig(session_init=sess_init,
                                   model=model,
                                   input_names=["feat", "seqlen"],
                                   output_names=["prediction"])

    predict_func = OfflinePredictor(predict_config)

    err_num = 0
    tot_num = 0
    if os.path.isfile(args.input_path):
        # input is a file
        result = predict_one(args.input_path, predict_func, None)
        label_filename = args.input_path.replace("png", "txt")
        if os.path.isfile(label_filename):
            with open(label_filename) as label_file:
                content = label_file.readlines()
                target = content[0]
            (cur_err, cur_len) = sequence_error_stat(target, result)
            err_num = err_num + cur_err
            tot_num = tot_num + cur_len
    if os.path.isdir(args.input_path):
        # input is a text file
        with open(args.input_path) as f:
            content = f.readlines()

        for idx, input_path in enumerate(content):
            result = predict_one(input_path, predict_func, idx + 1)
            label_filename = input_path.replace("png", "txt")
            if os.path.isfile(label_filename):
                with open(label_filename) as label_file:
                    content = label_file.readlines()
                    target = content[0]
                (cur_err, cur_len) = sequence_error_stat(target, result)
                err_num = err_num + cur_err
                tot_num = tot_num + cur_len

    logger.info("Character error rate is: " + str(err_num) + "/" + str(tot_num) + "(" + str(err_num * 1.0 / tot_num) + ")")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='path to the model file', required=True)
    parser.add_argument('--input_path', help='path to the input image')
    parser.add_argument('--test_path', help='path of the test file')

    args = parser.parse_args()
    predict(args)
