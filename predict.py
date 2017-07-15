import os
import numpy as np
from scipy import misc
import argparse
import cv2

from tensorpack import *

from train import Model
from mapper import *
from cfgs.config import cfg

import pdb

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

def predict_one(img_path, predict_func, idx):
    if cfg.input_channel == 1:
        img = misc.imread(img_path, 'L')
    else:
        img = misc.imread(img_path)
    if cfg.input_width != None:
        img = cv2.resize(img, (cfg.input_width, cfg.input_height))
    else:
        scale = cfg.input_height / img.shape[0]
        img = cv2.resize(img, fx=scale, fy=scale)
    seqlen = img.shape[1]
    if cfg.input_channel == 1:
        img = np.expand_dims(img, axis=2)
    img = np.expand_dims(img, axis=0)

    predictions = predict_func([img, [seqlen]])[0]

    mapper = Mapper()
    result = mapper.decode_output(predictions[0])
    # if idx == None:
    #     logger.info(img_path)
    #     logger.info(result)
    # else:
    #     logger.info(str(idx) + ": " + img_path)
    #     logger.info(str(idx) + ": " + result)
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
    if args.input_path is not None and os.path.isfile(args.input_path):
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
    if args.test_path is not None and os.path.isfile(args.test_path):
        # input is a text file
        with open(args.test_path) as f:
            content = f.readlines()

        lines = [e.strip() for e in content]

        for idx, input_path in enumerate(lines):
            result = predict_one(input_path, predict_func, idx + 1)
            ext = input_path.split('.')[1]
            label_filename = input_path.replace(ext, "txt")
            if os.path.isfile(label_filename):
                with open(label_filename) as label_file:
                    content = label_file.readlines()
                    target = content[0]
                (cur_err, cur_len) = sequence_error_stat(target, result)
                if cur_err > 0:
                    logger.info(input_path)
                    logger.info(target)
                    logger.info(result)
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
