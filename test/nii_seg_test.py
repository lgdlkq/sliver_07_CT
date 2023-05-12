#!usr/bin/env python3
# coding=utf-8
# @IDE    :PyCharm
# @File   :nii_seg_test.py
# @Date   :2019/5/9

import SimpleITK as sitk
import cv2
import numpy as np


def getSeg(source, mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.erode(mask, kernel)
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)
    for i in range(source.shape[0]):
        for j in range(source.shape[1]):
            result = cv2.pointPolygonTest(contours[0], (i, j), False)
            if result == -1:
                source[j, i] = 0
    return source


v_path = r'F:\sliver_07\volume-2.nii'  # segmentation volume
s_path = r'F:\sliver_07\segmentation-2.nii'  # segmentation volume
ds = sitk.ReadImage(s_path)
s_data = sitk.GetArrayFromImage(ds)
s_data[s_data == 2] = 1
print('------------read end segmentation---------------')
dv = sitk.ReadImage(v_path)
v_data = sitk.GetArrayFromImage(dv)
spacing = ds.GetSpacing()
print('------------read end volume---------------')
for i in range(v_data.shape[0]):
    if np.sum(s_data[i] == 1) == 0:
        v_data[i] = np.zeros(shape=[512, 512], dtype=np.float32)
        continue
    print(i)
    cv2.imshow('v',v_data[i])
    cv2.imshow('s',s_data[i])
    v_data[i] = getSeg(np.reshape(v_data[i], (512, 512)),
                       np.reshape(s_data[i], (512, 512)))
    cv2.imshow('d',np.reshape(v_data[i], (512, 512,1)))
    cv2.waitKey(0)
