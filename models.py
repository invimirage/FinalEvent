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
import json
from efficientnet_pytorch import EfficientNet


# ——————构造模型——————



class DoubleNet(nn.Module):
    def __init__(self, input_length, hidden_length, drop_out_rate):
        super(DoubleNet, self).__init__()
        self.quality_judger = nn.Sequential(
            nn.Linear(input_length, hidden_length),
            nn.ReLU(),
            nn.Dropout(drop_out_rate),
            nn.Linear(hidden_length, config.bin_number),
        )
        self.weight_judger = nn.Sequential(
            nn.BatchNorm1d(input_length),
            nn.Linear(input_length, hidden_length),
            nn.ReLU(),
            nn.Dropout(drop_out_rate),
            nn.Linear(hidden_length, 1),
            nn.Sigmoid(),
        )

    # seperates [0, 10, 30]递增序列，表示一个文本的分段在batch中的位置
    def forward(self, data, tag=None, separates:list=None, detail=False):
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
            if detail:
                print(weight_out[sta:end])
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

class SeparatedLSTM(nn.Module):
    def __init__(self, input_length, hidden_length, extra_length, layer_number, linear_hidden_length, drop_out_rate):
        super().__init__()
        self.input_length = input_length
        self.hidden_length = hidden_length
        self.lstm_layer_number = layer_number
        self.lstm = nn.LSTM(input_size=input_length, hidden_size=hidden_length, num_layers=layer_number, batch_first=True, dropout=drop_out_rate)
        self.fc = nn.Sequential(
            nn.Linear(hidden_length*layer_number+extra_length, linear_hidden_length),
            nn.ReLU(),
            nn.Dropout(drop_out_rate),
            nn.Linear(linear_hidden_length, config.bin_number),
        )

    # seperates [0, 10, 30]递增序列，表示一个文本的分段在batch中的位置
    def forward(self, text_data, extra_data, tag=None, detail=False):
        lstm_out, (hn, cn) = self.lstm(text_data)
        fc_input = torch.cat((hn.view(hn.shape[0], -1), extra_data), dim=1)
        output = self.fc(fc_input)
        out_probs = F.softmax(output, dim=1).detach()
        # print(tag)
        if tag is not None:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, tag)
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


# 模型输入为embeding之后的向量
class ScorerNet(nn.Module):
    def __init__(self, bert_path, hidden_length, grad_layer_name='encoder.layer.23.attention.self.query.weight', drop_out_rate=0.2):  # code_length为fc映射到的维度大小
        super().__init__()
        # embedding_dim = self.textExtractor.config.hidden_size
        self.bert_model = BertModel.from_pretrained(bert_path)
        with open(bert_path + 'config.json') as f:
            bert_config = json.load(f)
        bert_out_dim = bert_config['hidden_size']
        requires_grad = False
        for name, para in self.bert_model.named_parameters():
            if name == grad_layer_name:
                requires_grad = True
            para.requires_grad = requires_grad
        self.scorer = nn.Sequential(
            nn.Linear(bert_out_dim, hidden_length),
            nn.ReLU(),
            nn.Dropout(drop_out_rate),
            nn.Linear(hidden_length, config.bin_number),
        )

    def forward(self, input):
        bert_output = self.bert_model(
            input[0],
            token_type_ids=input[1],
            attention_mask=input[2]
        )
        last_encode = bert_output[0]
        scorer_output = self.scorer(last_encode)
        return scorer_output


class JudgeNet(nn.Module):
    def __init__(self, bert_path, grad_layer_name='encoder.layer.23.attention.self.query.weight', drop_out_rate=0.2):
        super().__init__()
        self.bert_model = BertModel.from_pretrained(bert_path)
        requires_grad = False
        for name, para in self.bert_model.named_parameters():
            if name == grad_layer_name:
                requires_grad = True
            para.requires_grad = requires_grad
        with open(bert_path + 'config.json') as f:
            bert_config = json.load(f)
        self.judger = nn.Sequential(
            nn.Linear(bert_config['hidden_size'], 1),
            nn.Sigmoid()
        )
    def forward(self, input, sep_points, labels=None):
        bert_output = self.bert_model(
            input[0],
            token_type_ids=input[1],
            attention_mask=input[2]
        )
        last_encode = bert_output[0]
        judge_output = self.judger(last_encode).squeeze(-1)
        slice_scores = []
        for i in range(len(sep_points)-1):
            sta = sep_points[i]
            end = sep_points[i+1]
            slice_scores.append(torch.mean(judge_output[:,sta:end], dim=1, keepdim=True))
        slice_scores_tensor = torch.cat(slice_scores, dim=1)
        if labels is not None:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(slice_scores_tensor, labels)
        else:
            loss = None
        return judge_output, loss

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
        feature_num = self.ec_model._fc.in_features
        self.ec_model._fc = nn.Linear(in_features=feature_num, out_features=config.bin_number, bias=True)
        # input = torch.randn((1, 3, 255, 255))
        # print(self.ec_model.extract_features(input).shape)
        # self.fc =

    def forward(self, input, tag=None):
        out = self.ec_model(input)
        # parsed_embed = self.deal_embed(embed).squeeze(-1).squeeze(-1)
        # out = self.fc(parsed_embed)
        out_probs = F.softmax(out, dim=1).detach()
        # print(tag)
        if tag is not None:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(out, tag)
        else:
            loss = 0
        return out_probs, loss
        # print(output.shape)

if __name__ == "__main__":
    tn = TextNet(128, "Bert/")
    for name, param in tn.textExtractor.named_parameters():
        if name == "pooler.dense.weight":
            print(param.shape)
        print(name)
