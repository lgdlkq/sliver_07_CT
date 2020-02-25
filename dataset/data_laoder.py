#!usr/bin/env python3
# coding=utf-8
# @IDE    :PyCharm
# @File   :data_laoder.py
# @author :雷国栋
# @Date   :2019/4/4

import os
import random
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from dataset.data_arguments import *
import configs
import numpy as np

random.seed(configs.seed)
np.random.seed(configs.seed)
torch.manual_seed(configs.seed)
torch.cuda.manual_seed_all(configs.seed)


def make_dataset(root):
    imgs = []
    n = len(os.listdir(root + 'seg'))
    for i in range(n):
        vol = os.path.join(root, "vol/%d.png" % i)
        seg = os.path.join(root, "seg/%d.png" % i)
        imgs.append((vol, seg))
    return imgs


class SliverDataset(Dataset):
    def __init__(self, root, mean=None, std=None, brightness=0.1, angle=15,
                 up_biase=0.04, left_biase=0.04):
        self.data = make_dataset(root)
        self.mean = mean
        self.std = std
        self.brightness = brightness
        self.angle = angle
        self.up_biase = up_biase
        self.left_biase = left_biase

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        vol_path, seg_path = self.data[idx]
        vol_img = Image.open(vol_path)
        seg_img = Image.open(seg_path)
        # seg_img = seg_img.resize((128, 128), Image.ANTIALIAS)
        tfs_parameter = []
        angle = random.uniform(-self.angle, self.angle)
        # H_possibility = random.random()
        # V_possibility = random.random()
        x_biase = random.uniform(-self.left_biase, self.left_biase)
        y_biase = random.uniform(-self.up_biase, self.up_biase)
        bright = random.uniform(-self.brightness, self.brightness)
        tfs_parameter.append(RandomRotation(angle))
        # tfs_parameter.append(RandomHorizontalFlip(H_possibility))
        # tfs_parameter.append(RandomVerticalFlip(V_possibility))
        tfs_parameter.append(RandomAffine(y_biase, x_biase))
        color_jitter = ColorJitter(bright)
        tfs_parameter.append(color_jitter)
        tfs_parameter.append(ToTensor())
        if not self.mean is None and not self.std is None:
            normal = Normalize(mean=self.mean, std=self.std)
            tfs_parameter.append(normal)
        transform = Compose(tfs_parameter)
        vol_img = transform(vol_img)
        seg_img = transform(seg_img)
        seg_img = ToPILImage()(seg_img)
        seg_img = seg_img.resize((128, 128), Image.ANTIALIAS)
        seg_img = ToTensor()(seg_img)
        return vol_img, seg_img


sliver = SliverDataset(configs.root)

train_size = int(configs.train_ratio * len(sliver))
valid_size = int(configs.valid_ratio * len(sliver))
test_size = len(sliver) - train_size - valid_size

train_dataset, valid_dataset, test_dataset = random_split(sliver, [train_size,
                                                                   valid_size,
                                                                   test_size])


def get_train_data(arg=False):
    train_dataloader = DataLoader(train_dataset, configs.batch_size,
                                  configs.shuffle)
    return train_dataloader


def get_valid_data(arg=False):
    valid_dataloader = DataLoader(valid_dataset, configs.batch_size,
                                  configs.shuffle)
    return valid_dataloader


def get_test_data(arg=False):
    test_dataloader = DataLoader(test_dataset, configs.batch_size,
                                 configs.shuffle)
    return test_dataloader
# train_dataloader = DataLoader(train_dataset, configs.batch_size, configs.shuffle)
# valid_dataloader = DataLoader(valid_dataset, configs.batch_size, configs.shuffle)
# test_dataloader = DataLoader(test_dataset, configs.batch_size, configs.shuffle)
