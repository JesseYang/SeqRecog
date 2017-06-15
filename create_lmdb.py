#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: create_lmdb.py
# Author: Jesse Yang <jesse.yang1985@gmail.com>
import sys
import os
from scipy import misc
import string
import numpy as np
import argparse

from tensorpack import *
from tensorpack.utils.argtools import memoized
from tensorpack.utils.stats import OnlineMoments

from mapper import Mapper

class TextDF(DataFlow):

    def __init__(self, dirname, dict_path, channel=1):
        self.dirname = dirname
        self.channel = channel
        self.filelists = [k for k in fs.recursive_walk(self.dirname)
                          if k.endswith('.png')]
        logger.info("Found {} png files ...".format(len(self.filelists)))

        self.mapper = Mapper(dict_path)

    def size(self):
        return len(self.filelists)

    def get_data(self):
        for filename in self.filelists:
            feat = misc.imread(filename, 'L')
            feat = np.expand_dims(feat, axis=2)
            label_filename = filename.replace("png", "txt")
            with open(label_filename) as label_file:
                content = label_file.readlines()
            yield [feat, self.mapper.encode_string(content[0])]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # subparsers = parser.add_subparsers(title='command', dest='command')
    parser.add_argument('--dataset', help='path to TRAIN or TEST directory', required=True)
    parser.add_argument('--dict_path', help='path to the dictionary file', default="dictionary_text")
    parser.add_argument('--channel', help='channels for input images', default=1)
    parser.add_argument('--db', help='output lmdb file', required=True)

    args = parser.parse_args()
    ds = TextDF(args.dataset, args.dict_path, args.channel)
    dftools.dump_dataflow_to_lmdb(ds, args.db)
