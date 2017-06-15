#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mapper.py
# Author: Jesse Yang <jesse.yang1985@gmail.com>

from tensorpack import BatchData
import numpy as np
from six.moves import range

__all__ = ['Mapper']


class Mapper(object):

    def __init__(self, dict_path):
        self.dict_path = dict_path
        self.alphabet2token = {}
        self.token2alphabet = {}

        with open(dict_path) as dict_file:
            lines = dict_file.read().splitlines()

        for lid, line in enumerate(lines):
            self.alphabet2token[line] = lid
            self.token2alphabet[lid] = line


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
