import logging

import time
import math
import matplotlib
import torch

matplotlib.rcParams['axes.unicode_minus'] = False


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since):
    now = time.time()
    s = now - since
    return '%s' % as_minutes(s)


def save_checkpoint(state, filename):
    torch.save(state, filename)
    logging.info('Saved the checkpoint')
