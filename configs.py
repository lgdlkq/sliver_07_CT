#!usr/bin/env python3
# coding=utf-8
# @IDE    :PyCharm
# @File   :configs.py
# @author :雷国栋
# @Date   :2019/4/3

import os
path = os.path.dirname(__file__)
root = "F:/sliver/"
train = path + "/data_split_list/trains.txt"
valid = path + "/data_split_list/valids.txt"
test = path + "/data_split_list/tests.txt"
model_name = "SliverNet"
weights = path + "/checkpoints/"
best_models = weights+"best/"
logs = path + '/logs/'
tf_logs = logs+'tf_logs/'
gpus = '0'

train_ratio = 0.7
valid_ratio = 0.15

epochs = 100
batch_size = 16
shuffle = True
num_workers = 0
img_height = 512
img_weight = 512
seed = 888
lr = 1e-4
final_lr = 1e-3
weight_decay = 0.0003
