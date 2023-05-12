#!usr/bin/env python3
# coding=utf-8
# @IDE    :PyCharm
# @File   :nt.py
# @Date   :2019/5/16

from vtk.util.vtkImageImportFromArray import *
import vtk
import SimpleITK as sitk
import numpy as np
import cv2

path = r'F:\sliver_07\segmentation-0.nii' #segmentation volume
ds = sitk.ReadImage(path)
data = sitk.GetArrayFromImage(ds)
spacing = ds.GetSpacing()
data = data[43:]
data = data.astype(np.uint8)
print(data.shape[0])
n_data = []
length = int(spacing[2])-1
for i in range(data.shape[0]-1):
    n_data.append(data[i])
    for j in range(length):
        temp = cv2.resize(data[i],(512+10*(j+1),512+10*(j+1)))
        temp = temp[5*(j+1):-5*(j+1),5*(j+1):-5*(j+1)]
        n_data.append(temp)
n_data = np.array(n_data)
print(n_data.shape)
