#!usr/bin/env python3
# coding=utf-8
# @IDE    :PyCharm
# @File   :mayav_test.py
# @Date   :2019/4/27

import SimpleITK as sitk
import cv2
import numpy as np
from mayavi import mlab

path = r'F:\sliver_07\segmentation-0.nii'  # segmentation-0.nii
ds = sitk.ReadImage(path)
image = sitk.GetArrayFromImage(ds)
# image[image == 2] = 1
# image = image.astype(dtype=np.float32) * 255
print(image.shape)
lz = image.shape[0]
lx = image.shape[1]
ly = image.shape[2]
print(lz)
print(lx)
print(ly)
spacing = ds.GetSpacing()  # x, y, z
print(spacing)

x, y, z = np.mgrid[-lz * spacing[2] / 2:lz * spacing[2] / 2:spacing[2],
          -lx * spacing[0] / 2:lx * spacing[0] / 2:spacing[0],
          -ly * spacing[1] / 2:ly * spacing[1] / 2:spacing[1]]
print('*' * 20)
print(x.shape)
print(y.shape)
print(z.shape)

mlab.figure(bgcolor=(1, 1, 1),size=(512,512))
mlab.contour3d(x,y,z,image,color=(0.53,0,0.098),name='seg',transparent=False,opacity=1.0)  #显示表面
mlab.show()
