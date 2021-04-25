#!/usr/bin/env python
# encoding: utf-8
'''
@author: zhangruifeng
@contact: zrf1999@pku.edu.cn
@file: model.py
@time: 2021/4/23 14:31
@github: local 16351726fa15c85f565b7d5fecdf320ea67a72ef
'''

from efficientnet_pytorch import EfficientNet
import config
import torch
import torch.nn as nn
import torch.functional as F

class PictureNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.ec_model = EfficientNet.from_name('efficientnet-b4')
        net_weight = torch.load('../Models/efficientnet-b4/efficientnet-b4-6ed6700e.pth')
        self.ec_model.load_state_dict(net_weight)
        grad = False
        for name, param in self.ec_model.named_parameters():
            if '_blocks.31' in name:
                grad = True
            param.requires_grad = grad
        self.deal_embed = nn.Sequential(
            self.ec_model._avg_pooling,
            self.ec_model._dropout,
        )
        self.fc = nn.Linear(1792, config.bin_number)
        # input = torch.randn((1, 3, 255, 255))
        # print(self.ec_model.extract_features(input).shape)
        # self.fc =

    def forward(self, input, tag=None):
        embed = self.ec_model.extract_features(input)
        parsed_embed = self.deal_embed(embed).squeeze(-1).squeeze(-1)
        out = self.fc(parsed_embed)
        out_probs = F.softmax(out, dim=1).detach()
        # print(tag)
        if tag is not None:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(out, tag)
        else:
            loss = 0
        return out_probs, loss
        # print(output.shape)

net = PictureNet()
input = torch.randn((1, 3, 255, 255))
net.forward(input=input)