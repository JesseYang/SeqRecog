import numpy as np
from scipy import misc
import cv2

from tensorpack import *

from mapper import Mapper
from cfgs.config import cfg

def batch_feature(feats):
    # pad to the longest in the batch
    maxlen = max([k.shape[1] for k in feats])
    bsize = len(feats)
    ret = np.zeros((bsize, feats[0].shape[0], maxlen, cfg.input_channel))
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

def get_imglist(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [ele.strip() for ele in content]
    return content

class Data(RNGDataFlow):
    def __init__(self, train_or_test, shuffle=True):
        assert train_or_test in ['train', 'test']
        fname_list = cfg.train_list if train_or_test == "train" else cfg.test_list
        self.train_or_test = train_or_test
        fname_list = [fname_list] if type(fname_list) is not list else fname_list

        self.imglist = []
        for fname in fname_list:
            self.imglist.extend(get_imglist(fname))
        self.shuffle = shuffle

        self.mapper = Mapper()

    def size(self):
        return len(self.imglist)

    def get_data(self):
        idxs = np.arange(len(self.imglist))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            img_path = self.imglist[k]
            label_path = img_path.split('.')[0] + ".txt"
            if cfg.input_channel == 1:
                img = misc.imread(img_path, 'L')
            else:
                img = misc.imread(img_path)
            if cfg.name == "plate":
                # augmente standard plate images with prob. of 0.7
                height, width, _ = img.shape
                if height == 184 and width == 577 and np.random.rand() > 0.2:
                    pad_ratio_w = 2
                    pad_ratio_h = 2
                    height_p = int(height * (1 + pad_ratio_h))
                    width_p = int(width * (1 + pad_ratio_w))

                    color = [int(np.random.rand() * 255), int(np.random.rand() * 255), int(np.random.rand() * 255)]
                    pad_img = np.tile(color, height_p * width_p)
                    pad_img = np.reshape(pad_img, (height_p, width_p, 3))
                    pad_img = pad_img.astype(np.uint8)

                    h_start = int(height * pad_ratio_h / 2)
                    w_start = int(width * pad_ratio_w / 2)
                    pad_img[h_start:h_start + height, w_start:w_start + width,:] = img

                    # random angle from -10 degree to 10 degree
                    angle = 20 * (np.random.rand() - 0.5)
                    scale = np.random.rand() * 0.2 - 0.1 + 1 
                    M = cv2.getRotationMatrix2D((width_p//2, height_p//2), angle, scale)
                    dst = cv2.warpAffine(pad_img, M, (width_p, height_p))

                    crop_h_ratio = 1.4
                    crop_w_ratio = 1.2
                    crop_h_start = int(height * pad_ratio_h / 2 - (crop_h_ratio - 1) / 2 * height + np.random.rand() * 30)
                    crop_w_start = int(width * pad_ratio_w / 2 - (crop_w_ratio - 1) / 2 * width + np.random.rand() * 30)
                    crop_h = int(height * crop_h_ratio)
                    crop_w = int(width * crop_w_ratio)
                    img = dst[crop_h_start:crop_h_start + crop_h, crop_w_start:crop_w_start + crop_w, :]
                if height == 184 and width == 577 and np.random.rand() > 0.2:
                    # blur the img
                    kernel_size = np.max([5, int(np.random.rand() * 20)])
                    kernel = np.ones((kernel_size, kernel_size),np.float32) / kernel_size / kernel_size
                    img = cv2.filter2D(img, -1, kernel)

            if cfg.input_width != None:
                img = cv2.resize(img, (cfg.input_width, cfg.input_height))
            else:
                scale = cfg.input_height / img.shape[0]
                img = cv2.resize(img, fx=scale, fy=scale)
            if cfg.input_channel == 1:
                feat = np.expand_dims(img, axis=2)
            else:
                feat = img
            with open(label_path) as f:
                content = f.readlines()
            label = self.mapper.encode_string(content[0])
            yield [feat, label]

class CTCBatchData(BatchData):

    def __init__(self, ds, batch_size, remainder=False):
        super(CTCBatchData, self).__init__(ds, batch_size, remainder)

    def get_data(self):
        itr = self.ds.get_data()
        for _ in range(self.size()):
            feats = []
            labels = []
            for b in range(self.batch_size):
                feat, label = next(itr)
                feats.append(feat)
                labels.append(label)
            # yield [feats, labs]
            batchfeat = batch_feature(feats)
            batchlabel = sparse_label(labels)
            seqlen = np.asarray([k.shape[1] for k in feats])
            yield [batchfeat, batchlabel[0], batchlabel[1], batchlabel[2], seqlen]
