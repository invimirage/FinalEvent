#!/usr/bin/env python
# encoding: utf-8
'''
@author: zhangruifeng
@contact: zrf1999@pku.edu.cn
@file: model.py
@time: 2021/3/21 10:58
@github: local 16351726fa15c85f565b7d5fecdf320ea67a72ef
'''
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
class TextNet(nn.Module):
    def __init__(self, hidden_length, bert_path):  # code_length为fc映射到的维度大小
        super(TextNet, self).__init__()
        self.textExtractor = BertModel.from_pretrained(bert_path)
        embedding_dim = self.textExtractor.config.hidden_size
        self.fcs = nn.Sequential(
            nn.Linear(embedding_dim, hidden_length),
            nn.ReLU(),
            nn.Linear(hidden_length, 1),
            nn.Sigmoid()
        )

    def forward(self, *args):
        output = self.textExtractor(args[0], token_type_ids=args[1],
                                    attention_mask=args[2])
        text_embeddings = output[0][:, 0, :]
        # output[0](batch size, sequence length, model hidden dimension)

        out = self.fcs(text_embeddings)
        return out

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

if __name__ == '__main__':
    tn = TextNet(128, 'Bert/')
    for name, param in tn.textExtractor.named_parameters():
        if name == 'pooler.dense.weight':
            print(param.shape)
        print(name)