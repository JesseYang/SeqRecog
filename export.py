#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train.py
# Author: Jesse Yang <jesse.yang1985@gmail.com>

import numpy as np
from scipy import misc
import argparse
import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants

from tensorpack import *

from train import Model
from mapper import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='path to the model file', required=True)
    parser.add_argument('--params_path', help='path to the params file', default="params_text.json")
    parser.add_argument('--output_names', help="names of the output operations, separated by ','", required=True)
    parser.add_argument('--output_path', help='path of the output model', default="model.pb")

    args = parser.parse_args()

    session_init = SaverRestore(args.model)
    model = Model(args.params_path, 1)


    config = ExportConfig(model=model,
                          session_init=session_init,
                          output_names=args.output_names.split(','))

    export = Export(config)

    export()
