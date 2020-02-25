#!usr/bin/env python3
# coding=utf-8
# @IDE    :PyCharm
# @File   :data_laoders.py
# @author :雷国栋
# @Date   :2019/4/5
import os
import random
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import configs
from dataset.data_arguments import *


def make_dataset(root, filename):
    imgs = []
    f = open(filename)
    name = f.readline()
    while name:
        vol = os.path.join(root, "vol/" + name.strip('\n'))
        seg = os.path.join(root, "seg/" + name.strip('\n'))
        imgs.append((vol, seg))
        name = f.readline()
    return imgs


class SliverDataset(Dataset):
    def __init__(self, root, filename=None, mean=None, std=None, arg=False,
                 brightness=0.1, angle=15, up_biase=0.04,
                 left_biase=0.04):
        self.data = make_dataset(root, filename)
        self.mean = mean
        self.std = std
        self.arg = arg
        self.brightness = brightness
        self.angle = angle
        self.up_biase = up_biase
        self.left_biase = left_biase

    def update_arg(self, arg):
        self.arg = arg

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        vol_path, seg_path = self.data[idx]
        vol_img = Image.open(vol_path)
        seg_img = Image.open(seg_path)
        tfs_parameter = []
        if self.arg:
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


train_sliver = SliverDataset(configs.root, configs.train)
valid_sliver = SliverDataset(configs.root, configs.valid)
test_sliver = SliverDataset(configs.root, configs.test)


def get_train_data(arg=False):
    train_sliver.update_arg(arg)
    train_dataloader = DataLoader(train_sliver, configs.batch_size,
                                  configs.shuffle)
    return train_dataloader


def get_valid_data(arg=False):
    valid_sliver.update_arg(arg)
    valid_dataloader = DataLoader(valid_sliver, configs.batch_size,
                                  configs.shuffle)
    return valid_dataloader


def get_test_data(arg=False):
    test_sliver.update_arg(arg)
    test_dataloader = DataLoader(test_sliver, configs.batch_size,
                                 configs.shuffle)
    return test_dataloader
