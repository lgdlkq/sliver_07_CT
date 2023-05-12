#!usr/bin/env python3
# coding=utf-8
# @IDE    :PyCharm
# @File   :utils.py
# @Date   :2019/4/4

import shutil
import torch
import os
import configs
import torch.nn as nn

def save_checkpoint(state, is_best,fold):
    filename = configs.weights + configs.model_name + os.sep +str(fold) + os.sep + "_checkpoint.pth.tar"
    torch.save(state, filename)
    if is_best:
        message = configs.best_models + configs.model_name+ os.sep +str(fold)  + os.sep + 'model_best.pth.tar'
        print("Get Better acc : %s from epoch %s saving weights to %s" % (state["best_precision"], str(state["epoch"]), message))
        writepath = '../logs/%s.txt'%configs.model_name
        mode = 'a' if os.path.exists(writepath) else 'w'
        with open(writepath, mode) as f:
            print("Get Better acc : %s from epoch %s saving weights to %s"%(state["best_precision"],str(state["epoch"]),message),file=f)
        shutil.copyfile(filename, message)

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss,self).__init__()
    def forward(self, *input):
        result = 1. - self.dice_coef(input[0], input[1])
        return result

    def dice_coef(self,y_true, y_pred):
        y_true_v = y_true.view(-1)
        y_pred_v = y_pred.view(-1)
        intersection = torch.sum(y_true_v*y_pred_v)
        return (2. * intersection + 1) / (torch.sum(y_true_v*y_true_v) + torch.sum(y_pred_v*y_pred_v) + 1)

class Accessory():
    def map_acc(self,y_true, y_pred):
        y_true_v = y_true.view(-1)
        y_pred_v = y_pred.view(-1)
        y_pred_v = torch.round(y_pred_v)
        x=torch.le(y_true_v,y_pred_v)
        y=torch.ge(y_true_v,y_pred_v)
        equel = torch.min(x,y).float()
        acc = torch.mean(equel)
        return acc

    def iou(self,y_trues, y_preds):
        y_true_v = y_trues.view(-1)
        y_pred_v = y_preds.view(-1)
        y_true_v = torch.round(y_true_v)
        y_pred_v = torch.round(y_pred_v)
        y_true_sum = torch.sum(y_true_v)
        y_pred_sum = torch.sum(y_pred_v)
        y_comb_v = torch.min(y_true_v, y_pred_v)
        y_comb_sum = torch.sum(y_comb_v)
        accs = y_comb_sum / (y_true_sum + y_pred_sum - y_comb_sum)
        return accs

    def ious(self,y_trues, y_preds):
        sums = 0
        length = y_trues.size(0)
        for i in range(y_trues.size(0)):
            y_true_v = y_trues[i].view(-1)
            y_pred_v = y_preds[i].view(-1)
            y_true_v = torch.round(y_true_v)
            y_pred_v = torch.round(y_pred_v)
            y_true_sum = torch.sum(y_true_v)
            y_pred_sum = torch.sum(y_pred_v)
            y_comb_v = torch.min(y_true_v, y_pred_v)
            y_comb_sum = torch.sum(y_comb_v)
            if y_true_sum==0 or y_true_sum + y_pred_sum - y_comb_sum==0:
                length -=1
                continue
            accs = y_comb_sum / (y_true_sum + y_pred_sum - y_comb_sum)
            sums += accs
        res = sums/length
        return res
