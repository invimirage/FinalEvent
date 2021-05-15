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
import os
from efficientnet_pytorch import EfficientNet
import math
import torch.nn.utils.rnn as rnn
from utils import *

# ——————构造模型——————


class MyModule(nn.Module):
    def __init__(self, name, requires_embed, hyperparams):
        super(MyModule, self).__init__()
        self.name = name
        self.requires_embed = requires_embed
        self.hyperparams = hyperparams


class DoubleNet(nn.Module):
    def __init__(self, input_length, hidden_length, drop_out_rate):
        super(DoubleNet, self).__init__()
        self.name = "DoubleNet"
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
    def forward(self, data, tag=None, separates: list = None, detail=False):
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


class BertWithCNN(MyModule):
    def __init__(self, bert_path, extra_length, hyperparams):  # code_length为fc映射到的维度大小
        super().__init__("BertWithCNN", False, hyperparams)
        hidden_length = hyperparams["hidden_length"]
        linear_hidden_length = hyperparams["linear_hidden_length"]
        grad_layer_name = hyperparams["grad_layer_name"]
        channels = hyperparams["channels"]
        drop_out_rate = hyperparams["drop_out_rate"]
        # embedding_dim = self.textExtractor.config.hidden_size
        self.bert_model = BertModel.from_pretrained(bert_path)
        with open(os.path.join(bert_path, "config.json")) as f:
            bert_config = json.load(f)
        bert_out_dim = bert_config["hidden_size"]
        requires_grad = False
        for name, para in self.bert_model.named_parameters():
            if name == grad_layer_name:
                requires_grad = True
            para.requires_grad = requires_grad
        self.pooler = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=(10, 10)),
            nn.ReLU(),
            # nn.Dropout(drop_out_rate),
            nn.AdaptiveAvgPool2d(hidden_length),
        )
        self.fcs = nn.Sequential(
            nn.Linear(
                hidden_length ** 2 * channels + extra_length, linear_hidden_length
            ),
            nn.ReLU(),
            nn.Dropout(drop_out_rate),
            nn.Linear(linear_hidden_length, config.bin_number),
        )

    def forward(self, input, extra_data, tag=None, detail=False):
        batch_size = extra_data.shape[0]
        bert_output = self.bert_model(
            input[0], token_type_ids=input[1], attention_mask=input[2]
        )
        last_encode = bert_output[0].unsqueeze(1)
        pooler_output = self.pooler(last_encode).view(batch_size, -1)
        fc_input = torch.cat((pooler_output, extra_data), dim=1)
        # fc_input = pooler_output
        output = self.fcs(fc_input)
        # print(output)
        out_probs = F.softmax(output, dim=1).cpu().detach()
        # print(tag)
        if tag is not None:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, tag)
        else:
            loss = 0
        return out_probs, loss


class SeparatedLSTM(MyModule):
    def __init__(self, input_length, extra_length, hyperparams):
        super().__init__("SeparatedLSTM", True, hyperparams)
        input_length = input_length
        hidden_length = hyperparams["hidden_length"]
        layer_number = hyperparams["layer_number"]
        linear_hidden_length = hyperparams["hidden_length"]
        drop_out_rate = hyperparams["drop_out_rate"]
        self.lstm = nn.LSTM(
            input_size=input_length,
            hidden_size=hyperparams["hidden_length"],
            num_layers=hyperparams["layer_number"],
            batch_first=True,
            dropout=hyperparams["drop_out_rate"],
        )
        self.fc = nn.Sequential(
            nn.Linear(
                hidden_length * layer_number + extra_length, linear_hidden_length
            ),
            nn.ReLU(),
            nn.Dropout(drop_out_rate),
            nn.Linear(linear_hidden_length, config.bin_number),
        )

    # seperates [0, 10, 30]递增序列，表示一个文本的分段在batch中的位置
    def forward(self, text_data, extra_data, tag=None, detail=False):
        # print(text_data)
        lstm_out, (hn, cn) = self.lstm(text_data)
        hn_batch_first = torch.transpose(hn, 0, 1).contiguous()
        fc_input = torch.cat(
            (hn_batch_first.view(hn_batch_first.shape[0], -1), extra_data), dim=1
        )
        # fc_input = hn_batch_first.view(hn_batch_first.shape[0], -1)
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
    def __init__(
        self,
        bert_path,
        hidden_length,
        grad_layer_name="encoder.layer.23.attention.self.query.weight",
        drop_out_rate=0.2,
    ):  # code_length为fc映射到的维度大小
        super().__init__()
        # embedding_dim = self.textExtractor.config.hidden_size
        self.bert_model = BertModel.from_pretrained(bert_path)
        with open(bert_path + "config.json") as f:
            bert_config = json.load(f)
        bert_out_dim = bert_config["hidden_size"]
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
            input[0], token_type_ids=input[1], attention_mask=input[2]
        )
        last_encode = bert_output[0]
        scorer_output = self.scorer(last_encode)
        return scorer_output


class JudgeNet(nn.Module):
    def __init__(
        self,
        bert_path,
        grad_layer_name="encoder.layer.23.attention.self.query.weight",
        drop_out_rate=0.2,
    ):
        super().__init__()
        self.bert_model = BertModel.from_pretrained(bert_path)
        requires_grad = False
        for name, para in self.bert_model.named_parameters():
            if name == grad_layer_name:
                requires_grad = True
            para.requires_grad = requires_grad
        with open(bert_path + "config.json") as f:
            bert_config = json.load(f)
        self.judger = nn.Sequential(
            nn.Linear(bert_config["hidden_size"], 1), nn.Sigmoid()
        )

    def forward(self, input, sep_points, labels=None):
        bert_output = self.bert_model(
            input[0], token_type_ids=input[1], attention_mask=input[2]
        )
        last_encode = bert_output[0]
        judge_output = self.judger(last_encode).squeeze(-1)
        slice_scores = []
        for i in range(len(sep_points) - 1):
            sta = sep_points[i]
            end = sep_points[i + 1]
            slice_scores.append(
                torch.mean(judge_output[:, sta:end], dim=1, keepdim=True)
            )
        slice_scores_tensor = torch.cat(slice_scores, dim=1)
        if labels is not None:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(slice_scores_tensor, labels)
        else:
            loss = None
        return judge_output, loss


class PictureNet(MyModule):
    def __init__(self, hyperparameters, out_dims=config.bin_number, extra_length=0):
        super().__init__(
            name="PictureNet", requires_embed=False, hyperparams=hyperparameters
        )
        linear_hidden_length = hyperparameters["linear_hidden_length"]
        self.ec_model = EfficientNet.from_name("efficientnet-b4")
        net_weight = torch.load(
            os.path.join(config.efficient_path, "efficientnet-b4-6ed6700e.pth")
        )
        self.ec_model.load_state_dict(net_weight)
        self.grad_layer_name = hyperparameters["grad_layer_name"]
        grad = False
        for name, param in self.ec_model.named_parameters():
            if self.grad_layer_name in name:
                grad = True
            param.requires_grad = grad
        feature_num = self.ec_model._fc.in_features
        self.fcs = nn.Sequential(
            nn.Linear(feature_num + extra_length, linear_hidden_length),
            nn.ReLU(),
            nn.Dropout(hyperparameters["drop_out_rate"]),
            nn.Linear(linear_hidden_length, out_dims),
        )
        #     in_features=feature_num, out_features=out_dims, bias=True
        # )
        # input = torch.randn((1, 3, 255, 255))
        # print(self.ec_model.extract_features(input).shape)
        # self.fc =

    def extract_features(self, img):
        img_feats = self.ec_model.extract_features(img)
        img_feats_pooled = self.ec_model._avg_pooling(img_feats).flatten(start_dim=1)
        return self.fcs(img_feats_pooled)

    def forward(self, img, extra=None, tag=None):
        img_feats = self.ec_model.extract_features(img)
        img_feats_pooled = self.ec_model._avg_pooling(img_feats).flatten(start_dim=1)
        if extra is not None:
            fc_input = torch.cat([img_feats_pooled, extra], dim=1)
        else:
            fc_input = img_feats_pooled
        out = self.fcs(fc_input)
        # x = self._dropout(x)
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

class VideoNetEmbed(MyModule):
    def __init__(self, extra_length, hyperparams):
        super().__init__(name="VideoNetEmbed", requires_embed=True, hyperparams=hyperparams)
        input_length = hyperparams["input_length"]
        hidden_length = hyperparams["hidden_length"]
        layer_number = hyperparams["layer_number"]
        linear_hidden_length = hyperparams["linear_hidden_length"]
        drop_out_rate = hyperparams["drop_out_rate"]
        self.lstm = nn.LSTM(
            input_size=input_length,
            hidden_size=hyperparams["hidden_length"],
            num_layers=hyperparams["layer_number"],
            batch_first=True,
            dropout=hyperparams["drop_out_rate"],
        )
        self.fc = nn.Sequential(
            nn.Linear(
                hidden_length * layer_number + extra_length, linear_hidden_length
            ),
            nn.ReLU(),
            nn.Dropout(drop_out_rate),
            nn.Linear(linear_hidden_length, config.bin_number),
        )

    # input为batch_size个list，每个list包含对应的视频帧数据，数据为cpu上的tensor
    def forward(
        self, video_input, extra_feats, tag=None):
        lengths = []
        for i in range(len(video_input)):
            lengths.append(video_input[i].shape[0])
        # print(len(video_data_input))
        # for i in video_data_input:
        #     print(i.shape)
        input_padded = rnn.pad_sequence(video_input, batch_first=True)
        input_packed = rnn.pack_padded_sequence(
            input_padded, lengths=lengths, batch_first=True, enforce_sorted=False
        )
        lstm_out, (hn, cn) = self.lstm(input_packed)
        hn_batch_first = torch.transpose(hn, 0, 1).contiguous()
        fc_input = torch.cat(
            (hn_batch_first.view(hn_batch_first.shape[0], -1), extra_feats), dim=1
        )
        # fc_input = hn_batch_first.view(hn_batch_first.shape[0], -1)
        output = self.fc(fc_input)
        out_probs = F.softmax(output, dim=1).detach()
        # print(tag)
        if tag is not None:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, tag)
        else:
            loss = 0
        return out_probs, loss

class VideoNet(MyModule):
    def __init__(self, extra_length, hyperparams):
        super().__init__(name="VideoNet", requires_embed=False, hyperparams=hyperparams)
        input_length = hyperparams["input_length"]
        pic_params = {}
        for key, param in hyperparams.items():
            if key.startswith("pic_"):
                pic_params[key.lstrip("pic_")] = param
        self.img_net = PictureNet(pic_params, out_dims=input_length, extra_length=0)
        hidden_length = hyperparams["hidden_length"]
        layer_number = hyperparams["layer_number"]
        linear_hidden_length = hyperparams["linear_hidden_length"]
        drop_out_rate = hyperparams["drop_out_rate"]
        self.lstm = nn.LSTM(
            input_size=input_length,
            hidden_size=hyperparams["hidden_length"],
            num_layers=hyperparams["layer_number"],
            batch_first=True,
            dropout=hyperparams["drop_out_rate"],
        )
        self.fc = nn.Sequential(
            nn.Linear(
                hidden_length * layer_number + extra_length, linear_hidden_length
            ),
            nn.ReLU(),
            nn.Dropout(drop_out_rate),
            nn.Linear(linear_hidden_length, config.bin_number),
        )

    # input为batch_size个list，每个list包含对应的视频帧数据，数据为cpu上的tensor
    def forward(
        self, video_input, extra_feats, tag=None, image_batch_size=100, device="cpu"
    ):
        frame_data = []
        sep_points = [0]
        for video_frames in video_input:
            # print(video_frames.shape)
            frame_data.append(video_frames)
            sep_points.append(sep_points[-1] + video_frames.shape[0])
        frame_data_flatten = torch.cat(frame_data, dim=0)
        n_batch = math.ceil(frame_data_flatten.shape[0] / image_batch_size)
        # print(frame_data_flatten.shape)
        img_embeds = []
        for i in range(n_batch):
            sta = i * image_batch_size
            end = (i + 1) * image_batch_size
            img_data = frame_data_flatten[sta:end]
            img_embed = self.img_net.extract_features(img_data.to(device))
            img_embeds.append(img_embed)
            # print(img_embed.shape)
        img_embeds_flatten = torch.cat(img_embeds)
        # print(img_embeds_flatten.shape)
        # print(sep_points)
        video_data_input = []
        lengths = []
        for i in range(len(video_input)):
            video_data_input.append(
                img_embeds_flatten[sep_points[i] : sep_points[i + 1]]
            )
            lengths.append(sep_points[i + 1] - sep_points[i])
        # print(len(video_data_input))
        # for i in video_data_input:
        #     print(i.shape)
        input_padded = rnn.pad_sequence(video_data_input, batch_first=True)
        input_packed = rnn.pack_padded_sequence(
            input_padded, lengths=lengths, batch_first=True, enforce_sorted=False
        )
        lstm_out, (hn, cn) = self.lstm(input_packed)
        hn_batch_first = torch.transpose(hn, 0, 1).contiguous()
        fc_input = torch.cat(
            (hn_batch_first.view(hn_batch_first.shape[0], -1), extra_feats), dim=1
        )
        # fc_input = hn_batch_first.view(hn_batch_first.shape[0], -1)
        output = self.fc(fc_input)
        out_probs = F.softmax(output, dim=1).detach()
        # print(tag)
        if tag is not None:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, tag)
        else:
            loss = 0
        return out_probs, loss



if __name__ == "__main__":
    tn = TextNet(128, "Bert/")
    for name, param in tn.textExtractor.named_parameters():
        if name == "pooler.dense.weight":
            print(param.shape)
        print(name)
