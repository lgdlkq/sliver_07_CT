#!usr/bin/env python3
# coding=utf-8
# @IDE    :PyCharm
# @File   :edge_test.py
# @author :雷国栋
# @Date   :2019/4/4
import torch
import cv2
import numpy as np
import PIL.Image as Image
import torchvision.transforms as tfs
import utils.utils as u

# import tensorflow as tf
# from keras import layers
# from keras import metrics
# import keras.backend as K
#
# smooth = 1.
#
# def dice_coef(y_true, y_pred):
#     y_true_f = layers.Flatten()(y_true)
#     y_pred_f = layers.Flatten()(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (2. * intersection + smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + smooth)
#
# def dice_coef_loss(y_true, y_pred):
#     result = 1.-dice_coef(y_true, y_pred)
#     return result
#
# def map_accuracy(y_true, y_pred):
#     y_true_f = layers.Flatten()(y_true)
#     y_pred_f = layers.Flatten()(y_pred)
#     acc = metrics.binary_accuracy(y_true_f,y_pred_f)
#     return acc

# src = cv2.imread('F:\sliver\seg/16.png')
# src1 = cv2.imread('F:\sliver\seg/15.png')
# img = Image.fromarray(src)
# img1 = Image.fromarray(src1)
# img = tfs.ToTensor()(img)
# img1 = tfs.ToTensor()(img1)
# r = u.map_accuracy(img,img1).numpy()
# print(r)
# t = u.iou(img,img1).numpy()
# print(t)

a=torch.Tensor([0.1,0.2,0.3,1.0,1.2,0.7,0.2])
b=torch.Tensor([0.1,0.52,0.3,0.4,0.51,0.6,0.8])
a=torch.round(a)
b=torch.round(b)
print(a)
print(b)
acc = u.Accessory()
print(torch.min(a,b))
print(acc.iou(a,b))

# a1 = tf.convert_to_tensor(np.reshape(src,(1,512,512,3)))
# a2 = tf.convert_to_tensor(np.reshape(src1,(1,512,512,3)))
# with tf.Session() as sess:
#     y_true_f = sess.run(layers.Flatten()(a1))
#     print(y_true_f.shape)
#     a3 = sess.run(map_accuracy(a1,a2))
#     print(a3)
