import numpy as np
from easydict import EasyDict as edict

cfg = edict()

cfg.name = 'plate'

cfg.input_height = 30
cfg.input_width = int(440 / 140 * cfg.input_height)
cfg.input_channel = 1

cfg.cnn = edict()
cfg.cnn.padding = "SAME"
cfg.cnn.channels = [32, 32, 32, 32, 64, 64]
cfg.cnn.kernel_heights = [3, 3, 3, 3, 3, 3]
cfg.cnn.kernel_widths = [3, 3, 3, 3, 3, 3]
cfg.cnn.with_bn = True

cfg.rnn = edict()
cfg.rnn.hidden_size = 660
cfg.rnn.hidden_layers_no = 3

cfg.weight_decay = 5e-4

cfg.dictionary = ["京", "津", "沪", "渝", "冀", "豫", "云", "辽", "黑", "湘",
                  "皖", "鲁", "新", "苏", "浙", "赣", "鄂", "桂", "甘", "晋",
                  "蒙", "陕", "吉", "闽", "贵", "粤", "青", "藏", "川", "宁",
                  "琼", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                  "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M",
                  "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

cfg.label_size = len(cfg.dictionary) + 1

cfg.train_list = [cfg.name + "_train.txt"]
cfg.test_list = cfg.name + "_test.txt"
