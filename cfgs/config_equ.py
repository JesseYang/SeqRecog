import numpy as np
from easydict import EasyDict as edict

cfg = edict()

cfg.name = 'equ'

cfg.input_height = 60
cfg.input_width = 180
cfg.input_channel = 1

cfg.cnn = edict()
cfg.cnn.padding = "VALID"
cfg.cnn.channels = [32, 32, 32, 32, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256]
cfg.cnn.kernel_heights = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
cfg.cnn.kernel_widths = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
cfg.cnn.with_bn = True

cfg.rnn = edict()
cfg.rnn.hidden_size = 660
cfg.rnn.hidden_layers_no = 2

cfg.weight_decay = 5e-4

cfg.dictionary = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                  "+", "-", "*", "(", ")"]

cfg.label_size = len(cfg.dictionary) + 1

cfg.train_list = [cfg.name + "_train.txt"]
cfg.test_list = cfg.name + "_test.txt"
