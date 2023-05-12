#!usr/bin/env python3
# coding=utf-8
# @IDE    :PyCharm
# @File   :net.py
# @Date   :2019/4/5
from collections import OrderedDict

import cv2
import numpy as np
import torch
from torch import nn
import configs

__all__ = ['SliverNet']

skip_connections = []


class DenseLayer(nn.Sequential):
    def __init__(self, i, num_input_features, growth_rate, bn_size, drop_rate):
        super(DenseLayer, self).__init__()
        self.add_module('dl' + str(i) + '_norm1',
                        nn.BatchNorm2d(num_input_features)),
        self.add_module('dl' + str(i) + '_relu1', nn.ReLU(inplace=True)),
        self.add_module('dl' + str(i) + '_conv2',
                        nn.Conv2d(num_input_features, bn_size*growth_rate,
                                  kernel_size=(3,3), stride=1, padding=1,
                                  bias=True)),
        self.add_module('dl' + str(i) + '_drop', nn.Dropout(drop_rate, True))

    def forward(self, x):
        new_features = super(DenseLayer, self).forward(x)
        dl_tensor = torch.cat([x, new_features], 1)
        return dl_tensor


class DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
                 drop_rate, skips):
        super(DenseBlock, self).__init__()
        self.features = 1
        self.skips = skips
        for i in range(num_layers):
            layer = DenseLayer(i, num_input_features + i * growth_rate,
                               growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % i, layer)
            self.features = num_input_features + (i + 1) * growth_rate

    def forward(self, x):
        new_features = super(DenseBlock, self).forward(x)
        if self.skips:
            skip_connections.append(new_features)
        return new_features


class TransitionDown(nn.Sequential):
    def __init__(self, i, num_input_features, num_output_features, drop_rate):
        super(TransitionDown, self).__init__()
        self.add_module('td' + str(i) + '_norm',
                        nn.BatchNorm2d(num_input_features))
        self.add_module('td' + str(i) + '_relu', nn.ReLU(inplace=True))
        self.add_module('td' + str(i) + '_conv',
                        nn.Conv2d(num_input_features, num_output_features,
                                  kernel_size=1, stride=1, bias=True))
        self.add_module('td' + str(i) + '_drop', nn.Dropout(drop_rate, True))
        self.add_module('td' + str(i) + '_pool',
                        nn.MaxPool2d(kernel_size=(2,2), stride=2))


class TransitionUp(nn.Sequential):
    def __init__(self, i, num_input_features, num_output_features):
        super(TransitionUp, self).__init__()
        self.add_module('tu' + str(i) + '_tconv',
                        nn.ConvTranspose2d(num_input_features,
                                           num_output_features,
                                           kernel_size=(2, 2),
                                           stride=2))
        self.i = i

    def forward(self, x):
        new_features = super(TransitionUp, self).forward(x)
        dl_tensor = torch.cat([new_features, skip_connections[self.i]], 1)
        return dl_tensor


class SliverNet(nn.Module):
    def __init__(self, n_layers=3, growth_rate=24,
                 layer_per_block=[3, 5, 7, 9, 7, 5, 3],
                 nb_features=48, drop_rate=0.2):
        super(SliverNet, self).__init__()

        self.prepare = nn.Sequential(OrderedDict([
            (
            'conv0', nn.Conv2d(1, 48, kernel_size=(7, 7), stride=2, padding=3)),
            ('norm0', nn.BatchNorm2d(48)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=(2, 2)))]))
        self.modelD = nn.Sequential()
        for i in range(n_layers):
            if i == 0:
                num_input_features = nb_features
            else:
                num_input_features = self.modelD.__getattr__(
                    'db' + str(i - 1)).features
            self.modelD.add_module('db' + str(i),
                                   DenseBlock(num_layers=layer_per_block[i],
                                              num_input_features=num_input_features,
                                              bn_size=1,
                                              growth_rate=growth_rate,
                                              drop_rate=drop_rate,
                                              skips=True))
            self.modelD.add_module('td' + str(i),
                                   TransitionDown(0, self.modelD.__getattr__(
                                       'db' + str(i)).features,
                                                  self.modelD.__getattr__(
                                                      'db' + str(i)).features,
                                                  drop_rate))

        self.center = DenseBlock(layer_per_block[n_layers],
                                 self.modelD.__getitem__(-2).features,
                                 1, growth_rate, drop_rate, False)

        self.modelU = nn.Sequential()
        for i in range(n_layers):
            iter = n_layers + i + 1
            keep_nb_features = growth_rate * layer_per_block[n_layers + i]
            skips_i = keep_nb_features + growth_rate * (
                    sum(layer_per_block[iter:]) + 2)
            if i == 0:
                num_input_features = self.center.features
            else:
                num_input_features = self.modelU.__getitem__(-1).features
            self.modelU.add_module('tu' + str(i),
                                   TransitionUp(i, num_input_features,
                                                keep_nb_features))
            self.modelU.add_module('db' + str(i),
                                   DenseBlock(num_layers=layer_per_block[iter],
                                              num_input_features=skips_i,
                                              bn_size=1,
                                              growth_rate=growth_rate,
                                              drop_rate=drop_rate,
                                              skips=False))

        self.forecast = nn.Sequential(OrderedDict([
            ('norml', nn.BatchNorm2d(self.modelU.__getitem__(-1).features)),
            ('relu', nn.ReLU(inplace=True)),
            ('conv', nn.Conv2d(self.modelU.__getitem__(-1).features, 1,
                               kernel_size=(1, 1))),
            ('sigmoid', nn.Sigmoid())]))

    def forward(self, x):
        global skip_connections
        skip_connections = []
        x = self.prepare(x)
        x = self.modelD(x)
        x = self.center(x)
        skip_connections = list(reversed(skip_connections))
        x = self.modelU(x)
        # print(x.size())
        out = self.forecast(x)
        return out


if __name__ == '__main__':
    from torchsummary import summary

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    s = SliverNet().to(device)
    summary(model=s, input_size=(1, 512, 512))

    # src = cv2.imread('G:/PythonTrainFaile/u_net_liver_data/train/000.png',
    #                  cv2.COLOR_BGR2GRAY)
    # src = np.reshape(src[:, :, 1], (1, 1, 512, 512))
    # x = torch.Tensor(src)
    # s = SliverNet()
    # s(x)

    # torch.save(s.state_dict(), 'd_unet_weights.pth')
