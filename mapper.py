#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mapper.py
# Author: Jesse Yang <jesse.yang1985@gmail.com>

from tensorpack import BatchData
import numpy as np
from six.moves import range

from cfgs.config import cfg

class Mapper(object):

    def __init__(self):
        self.alphabet2token = {}
        self.token2alphabet = {}

        for c_id, char in enumerate(cfg.dictionary):
            self.alphabet2token[char] = c_id
            self.token2alphabet[c_id] = char


    def encode_string(self, line):
        label = []
        for char in line:
            label.append(self.alphabet2token[char])
        return label

    def decode_output(self, predictions):
        line = ""
        for label in predictions:
            line += self.token2alphabet[label]
        return line
