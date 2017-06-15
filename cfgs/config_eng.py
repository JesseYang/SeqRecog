import numpy as np
from easydict import EasyDict as edict

cfg = edict()

cfg.name = 'english'

cfg.input_height = 10
cfg.input_channel = 1
cfg.label_size = 57

cfg.cnn = edict()
cfg.cnn.padding = "SAME"
cfg.cnn.channels = [32, 32, 32, 32, 64, 64]
cfg.cnn.kernel_heights = [3, 3, 3, 3, 3, 3]
cfg.cnn.kernel_widths = [3, 3, 3, 3, 3, 3]
cfg.cnn.with_bn = True

cfg.rnn = edict()
cfg.rnn.hidden_size = 660
cfg.rnn.hidden_layers_no = 3

cfg.label_size = 57

cfg.weight_decay = 5e-4

cfg.dictionary = [" ", "\"", "$", "%", "&", "'", "(", ")", "*",
                  "-", ".", "/", "0", "1", "2", "3", "4", "5",
                  "6", "7", "8", "9", ":", "<", ">", "?", "[",
                  "]", "a", "b", "c", "d", "e", "f", "g", "h",
                  "i", "j", "k", "l", "m", "n", "o", "p", "q",
                  "r", "s", "t", "u", "v", "w", "x", "y", "z",
                  "{", "}"]

cfg.train_list = [cfg.name + "_train.txt"]
cfg.test_list = cfg.name + "_test.txt"
