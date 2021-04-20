#!/usr/bin/env python
# encoding: utf-8
"""
@author: zhangruifeng
@contact: zrf1999@pku.edu.cn
@file: model.py
@time: 2021/3/21 10:58
@github: local 16351726fa15c85f565b7d5fecdf320ea67a72ef
"""
from transformers import BertModel, BertConfig, BertTokenizer
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, BertConfig
import pandas as pd
import numpy as np
import config


# ——————构造模型——————
# 模型输入为embeding之后的向量
class TextNet(nn.Module):
    def __init__(self, hidden_length, bert_path):  # code_length为fc映射到的维度大小
        super(TextNet, self).__init__()
        # embedding_dim = self.textExtractor.config.hidden_size
        self.layers = nn.Sequential(
            # nn.AdaptiveAvgPool2d(hidden_length),
            # nn.ReLU(),
            nn.Linear(hidden_length, config.bin_number),
        )

    def forward(self, data, tag=None):
        out = self.layers(data)
        if tag is not None:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(out, tag)
        else:
            loss = 0
        return out, loss


class DoubleNet(nn.Module):
    def __init__(self, input_length, hidden_length, drop_out_rate):
        super(DoubleNet, self).__init__()
        self.quality_judger = nn.Sequential(
            nn.Linear(input_length, hidden_length),
            nn.Dropout(drop_out_rate),
            nn.ReLU(),
            nn.Linear(hidden_length, config.bin_number),
        )
        self.weight_judger = nn.Sequential(
            nn.BatchNorm1d(input_length),
            nn.Linear(input_length, hidden_length),
            nn.Dropout(drop_out_rate),
            nn.ReLU(),
            nn.Linear(hidden_length, 1),
            nn.Sigmoid(),
        )

    # seperates [0, 10, 30]递增序列，表示一个文本的分段在batch中的位置
    def forward(self, data, tag=None, separates=None):
        weight_out = self.weight_judger(data)
        # print(1, weight_out)
        quality_out = self.quality_judger(data) * weight_out
        # print(2, quality_out)
        num_texts = len(separates) - 1
        # print(3, num_texts)
        output = []
        for i in range(num_texts):
            sta = separates[i]
            end = separates[i + 1]
            # log_prob = torch.log(quality_out[sta:end] + 1e-20)
            probs = quality_out[sta:end]
            mean_prob = torch.mean(probs, dim=0, keepdim=True)
            # print(mean_prob.shape)
            output.append(mean_prob)
        # print(output)
        output_tensor = torch.cat(output)
        # print(output_tensor.shape)
        out_probs = F.softmax(output_tensor, dim=1).detach()
        # print(tag)
        if tag is not None:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output_tensor, tag)
        else:
            loss = 0
        return out_probs, loss


class deal_embed(nn.Module):
    def __init__(self, bert_out_length, hidden_length):  # code_length为fc映射到的维度大小
        super(deal_embed, self).__init__()
        self.fcs = nn.Sequential(
            nn.Linear(bert_out_length, config.bin_number),
            # nn.ReLU(),
            # nn.Linear(hidden_length, 1),
        )

    def forward(self, data, tag=None):
        out = self.fcs(data)
        if tag is not None:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(out, tag)
        else:
            loss = 0
        return out, loss


if __name__ == "__main__":
    tn = TextNet(128, "Bert/")
    for name, param in tn.textExtractor.named_parameters():
        if name == "pooler.dense.weight":
            print(param.shape)
        print(name)
