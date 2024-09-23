import datetime
import os
import time
import torch
from torch.utils.data import DataLoader

import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
import sys
from torch.cuda import amp
import smodels
import argparse
from spikingjelly.clock_driven import functional
from spikingjelly.datasets import cifar10_dvs
import math
# from loss import *
import torch.nn as nn

_seed_ = 2020
import random
random.seed(2020)

torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import numpy as np
np.random.seed(_seed_)

from torch.utils.data import Dataset

class ConcatDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

        # Assuming both datasets have the same length
        assert len(dataset1) == len(dataset2), "Datasets must have the same length"

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, idx):
        # Get samples and labels from both datasets
        sample1, label1 = self.dataset1[idx]
        sample2, label2 = self.dataset2[idx]

        assert label1 == label2

        return sample1, sample2, label1

def split_to_train_test_set(train_ratio: float, origin_dataset: torch.utils.data.Dataset, num_classes: int, random_split: bool = False):
    '''
    :param train_ratio: split the ratio of the origin dataset as the train set
    :type train_ratio: float
    :param origin_dataset: the origin dataset
    :type origin_dataset: torch.utils.data.Dataset
    :param num_classes: total classes number, e.g., ``10`` for the MNIST dataset
    :type num_classes: int
    :param random_split: If ``False``, the front ratio of samples in each classes will
            be included in train set, while the reset will be included in test set.
            If ``True``, this function will split samples in each classes randomly. The randomness is controlled by
            ``numpy.randon.seed``
    :type random_split: int
    :return: a tuple ``(train_set, test_set)``
    :rtype: tuple
    '''
    label_idx = []
    for i in range(num_classes):
        label_idx.append([])

    for i, item in enumerate(origin_dataset):
        y = item[2]
        if isinstance(y, np.ndarray) or isinstance(y, torch.Tensor):
            y = y.item()
        label_idx[y].append(i)
    train_idx = []
    test_idx = []
    if random_split:
        for i in range(num_classes):
            np.random.shuffle(label_idx[i])

    for i in range(num_classes):
        pos = math.ceil(label_idx[i].__len__() * train_ratio)
        train_idx.extend(label_idx[i][0: pos])
        test_idx.extend(label_idx[i][pos: label_idx[i].__len__()])

    return torch.utils.data.Subset(origin_dataset, train_idx), torch.utils.data.Subset(origin_dataset, test_idx)

if __name__ == '__main__':
    origin_set = cifar10_dvs.CIFAR10DVS(root="/nvme/cifar-10-dvs", data_type='frame', frames_number=8, split_by='number')
    origin_set_teacher = cifar10_dvs.CIFAR10DVS(root="/nvme/cifar-10-dvs", data_type='frame', frames_number=16,
                                        split_by='number')

    concat_dataset = ConcatDataset(origin_set, origin_set_teacher)
    train_set, test_set = split_to_train_test_set(0.9, concat_dataset, 10)
    torch.save(train_set, 'dts_cache/train_set_8_16.pt')
    torch.save(test_set, 'dts_cache/test_set_8_16.pt')