#!usr/bin/env python3
# coding=utf-8
# @IDE    :PyCharm
# @File   :dataargument_test.py
# @Date   :2019/4/3
'''
重写pytorch的部分数据提升方法，便于实现vol和seg_mask做相同的数据提升操作
'''
from __future__ import division
import torch
import math
import numpy as np
import types

from PIL import Image

from dataset import functional as F
from torch.nn import functional as Ff

__all__ = ["Compose", "ToTensor", "ToPILImage", "Normalize", "RandomRotation",
        "RandomHorizontalFlip", "RandomVerticalFlip",  "ColorJitter","RandomAffine"]

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

class ToTensor(object):
    def __call__(self, pic):
        return F.to_tensor(pic)

class ToPILImage(object):
    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, pic):
        return F.to_pil_image(pic, self.mode)

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return F.normalize(tensor, self.mean, self.std)

class RandomRotation(object):
    def __init__(self,angle,resample=False, expand=False, center=None):
        self.angle = angle
        self.resample = resample
        self.expand = expand
        self.center = center

    def __call__(self, img):
        return F.rotate(img, self.angle, self.resample, self.expand, self.center)

class RandomHorizontalFlip(object):
    def __init__(self,probability):
        self.probability = probability
    def __call__(self, img):
        if self.probability < 0.5:
            return F.hflip(img)
        return img

class RandomVerticalFlip(object):
    def __init__(self,probability):
        self.probability = probability
    def __call__(self, img):
        if self.probability < 0.5:
            return F.vflip(img)
        return img

class Lambda(object):
    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        transforms = []
        if brightness > 0:
            brightness_factor = np.random.uniform(max(0, 1 - brightness), 1 + brightness)
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast > 0:
            contrast_factor = np.random.uniform(max(0, 1 - contrast), 1 + contrast)
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation > 0:
            saturation_factor = np.random.uniform(max(0, 1 - saturation), 1 + saturation)
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue > 0:
            hue_factor = np.random.uniform(-hue, hue)
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        np.random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, img):
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return transform(img)

class RandomAffine(object):
    def __init__(self,up_biase,left_biase):
        self.theta = torch.tensor([
            [1, 0, left_biase],  # 左右移动
            [0, 1, up_biase] ] , dtype=torch.float)# 上下移动

    def __call__(self,img):
        img_torch = ToTensor()(img)
        grid = Ff.affine_grid(self.theta.unsqueeze(0), img_torch.unsqueeze(0).size())
        output = Ff.grid_sample(img_torch.unsqueeze(0), grid)
        new_img_torch = output[0]
        img = ToPILImage()(new_img_torch)
        return img
