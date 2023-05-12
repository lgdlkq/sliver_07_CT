#!usr/bin/env python3
# coding=utf-8
# @IDE    :PyCharm
# @File   :data_split.py
# @Date   :2019/4/5

import random

import os
from sklearn.model_selection import train_test_split,KFold
import configs

def make_dataset(root):
    vol_imgs = []
    seg_imgs = []
    n = len(os.listdir(root + 'neg_seg'))
    for i in range(n):
        vol =  "%d.png" % i
        seg =  "%d.png" % i
        vol_imgs.append(vol)
        seg_imgs.append(seg)
    return vol_imgs,seg_imgs

vol_imgs,seg_imgs = make_dataset(configs.root)

#10折交叉验证方式划分
# split_fold = KFold(n_splits=10,shuffle=True)
# folds = split_fold.split(vol_imgs,seg_imgs)
# folds = list(folds)
# print(len(folds[0][0]))
# print(len(folds[0][1]))
# for train_index , test_index in folds:
#     print('train_index:%s , test_index: %s ' %(train_index,test_index))

train_vol, valid_vol, train_seg, valid_seg = train_test_split(vol_imgs, seg_imgs,
                        shuffle=True,test_size=0.3,random_state=random.randint(0, 150))

test_vol, valid_vol, test_seg, valid_seg = train_test_split(valid_vol, valid_seg,
                        shuffle=True, test_size=0.5,random_state=random.randint(0, 150))

with open("../data_split_list/neg_trains.txt",'a+') as f:
    for i in train_seg:
        f.write(i+'\n')

with open("../data_split_list/neg_tests.txt",'a+') as f:
    for i in test_seg:
        f.write(i+'\n')

with open("../data_split_list/neg_valids.txt",'a+') as f:
    for i in valid_seg:
        f.write(i+'\n')
