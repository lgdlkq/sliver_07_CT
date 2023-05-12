#!usr/bin/env python3
# coding=utf-8
# @IDE    :PyCharm
# @File   :data_ready.py
# @Date   :2019/4/21

import random
import torch
from PIL import Image
from dataset.data_arguments import *
import configs
import numpy as np

random.seed(configs.seed)
np.random.seed(configs.seed)
torch.manual_seed(configs.seed)
torch.cuda.manual_seed_all(configs.seed)


class ReadyData():
    def __init__(self, data=None, mean=None, std=None, arg=False,
                 brightness=0.1, angle=15, up_biase=0.04,
                 left_biase=0.04):
        self.data = data
        self.mean = mean
        self.std = std
        self.arg = arg
        self.brightness = brightness
        self.angle = angle
        self.up_biase = up_biase
        self.left_biase = left_biase

    def update_arg(self, arg):
        self.arg = arg

    def update_data(self, data):
        self.data = data

    def __len__(self):
        if self.data is None:
            return 0
        return len([self.data])

    def getitem(self):
        img = Image.fromarray(self.data)
        tfs_parameter = []
        if self.arg:
            angle = random.uniform(-self.angle, self.angle)
            x_biase = random.uniform(-self.left_biase, self.left_biase)
            y_biase = random.uniform(-self.up_biase, self.up_biase)
            bright = random.uniform(-self.brightness, self.brightness)
            tfs_parameter.append(RandomRotation(angle))
            tfs_parameter.append(RandomAffine(y_biase, x_biase))
            color_jitter = ColorJitter(bright)
            tfs_parameter.append(color_jitter)
        tfs_parameter.append(ToTensor())
        if not self.mean is None and not self.std is None:
            normal = Normalize(mean=self.mean, std=self.std)
            tfs_parameter.append(normal)
        transform = Compose(tfs_parameter)
        img = transform(img)
        return img.unsqueeze(0)
