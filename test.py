#!usr/bin/env python3
# coding=utf-8
# @IDE    :PyCharm
# @File   :edge_test.py
# @Date   :2019/4/2

from operation import Operater
from dataset.data_ready import ReadyData
import cv2
import numpy as np

op = Operater()
op.load_model()
r_dset = ReadyData()
for i in range(617,633):
    print(i)
    src = cv2.imread(r"F:\sliver\vol/" + str(i)+".png",cv2.IMREAD_UNCHANGED)
    r_dset.update_data(src)
    seg = cv2.imread(r"F:\sliver\seg/" + str(i)+".png",cv2.IMREAD_UNCHANGED)
    cv2.imshow('a',seg)
    op.predict_pic(r_dset.getitem().float())
