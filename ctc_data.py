#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: ctcdata.py
# Author: Jesse Yang <jesse.yang1985@gmail.com>

from tensorpack import BatchData
import numpy as np
from six.moves import range

__all__ = ['CTCBatchData']


def batch_feature(feats):
    # pad to the longest in the batch
    maxlen = max([k.shape[1] for k in feats])
    bsize = len(feats)
    ret = np.zeros((bsize, feats[0].shape[0], maxlen, 1))
    for idx, feat in enumerate(feats):
        ret[idx, :, :feat.shape[1]] = feat
    return ret


def sparse_label(labels):
    maxlen = max([len(k) for k in labels])
    shape = [len(labels), maxlen]   # bxt
    indices = []
    values = []
    for bid, lab in enumerate(labels):
        for tid, c in enumerate(lab):
            indices.append([bid, tid])
            values.append(c)
    indices = np.asarray(indices)
    values = np.asarray(values)
    return (indices, values, shape)


class CTCBatchData(BatchData):

    def __init__(self, ds, batch_size, remainder=False):
        super(CTCBatchData, self).__init__(ds, batch_size, remainder)

    def get_data(self):
        itr = self.ds.get_data()
        for _ in range(self.size()):
            feats = []
            labs = []
            for b in range(self.batch_size):
                feat, lab = next(itr)
                feats.append(feat)
                labs.append(lab)
            # yield [feats, labs]
            batchfeat = batch_feature(feats)
            batchlab = sparse_label(labs)
            seqlen = np.asarray([k.shape[1] for k in feats])
            yield [batchfeat, batchlab[0], batchlab[1], batchlab[2], seqlen]
