"""
Utility Functions

Provides common utility functions including average meter for tracking
training metrics and directory creation helper.

Original Author: Haiyang Mei (mhy666@mail.dlut.edu.cn)
Reference: CVPR2021_PFNet
"""
import numpy as np
import os


class AvgMeter(object):
    """Running average meter for tracking scalar values during training."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def check_mkdir(dir_name):
    """Create directory if it does not exist."""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def _sigmoid(x):
    """Compute sigmoid function."""
    return 1 / (1 + np.exp(-x))