#!usr/bin/env python3
# coding=utf-8
# @IDE    :PyCharm
# @File   :edge_test.py
# @Date   :2019/4/12

import cv2
import numpy as np

v_path = r'G:\PythonTrainFaile\u_net_liver_data\train\063.png'
s_path = r'G:\PythonTrainFaile\u_net_liver_data\train\063_mask.png'

src_v = cv2.imread(v_path)
src = cv2.imread(s_path,cv2.COLOR_BGR2GRAY)
print(src.shape)
org = src_v[:,:,0].copy()
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
mask = cv2.erode(src,kernel)
# edge = cv2.Canny(mask,30,100)
# edge = cv2.dilate(edge,kernel)

# org = cv2.cvtColor(org,cv2.COLOR_GRAY2BGR)
# org[edge!=0] = [0,127,255]

_,contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for i in range(512):
    for j in range(512):
        result = cv2.pointPolygonTest(contours[0], (i, j), False)
        if result == -1:
            org[j,i] = 0
cv2.imshow('v',src_v)
# cv2.imshow('e',edge)
cv2.imshow('r',org)
cv2.waitKey(0)
