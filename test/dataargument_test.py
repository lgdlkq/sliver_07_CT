#!usr/bin/env python3
# coding=utf-8
# @IDE    :PyCharm
# @File   :dataargument_test.py
# @Date   :2019/4/3
import random
from PIL import Image
from dataset.data_arguments import *
import numpy as np

img_path = "../resource/1.png"
img=Image.open(img_path)

# probile = random.random()
# print(probile)
# img = RandomHorizontalFlip(probile)(img)
# img = RandomVerticalFlip(probile)(img)

# angle = random.uniform(-20,20)
# img = RandomRotation(angle)(img)

# img = ColorJitter(0.7)(img)

# x=random.uniform(-0.3,0.3)
# y=random.uniform(-0.3,0.3)
# print(x,'  ',y)
# img = RandomAffine(x,y)(img)

img = img.resize((480,480),Image.ANTIALIAS)
img.show()
