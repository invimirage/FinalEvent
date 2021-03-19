#!/usr/bin/env python
# encoding: utf-8
'''
@author: ruifengzhang
@software:XXXX
@file: text_scorer.py
@time: 2021/3/15 10:21
@desc:
'''

import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel, BertTokenizer, BertConfig, BertAdam
import pandas as pd
import numpy as np
from tqdm import tqdm
# from torch.utils.data import *

bert_path = "Bert/"
tokenizer = BertTokenizer(vocab_file=bert_path + "vocab.txt")  # 初始化分词器

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)  # /bert_pretrain/
        for param in self.bert.parameters():
            param.requires_grad = True  # 每个参数都要 求梯度
        self.fc = nn.Linear(768, 10)   # 768 -> 2

    def forward(self, x):
        context = x[0]  # 输入的句子   (ids, seq_len, mask)
        types = x[1]
        mask = x[2]  # 对padding部分进行mask，和句子相同size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, token_type_ids=types,
                              attention_mask=mask,
                              output_all_encoded_layers=False) # 控制是否输出所有encoder层的结果
        out = self.fc(pooled)   # 得到10分类
        return out

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(DEVICE)
print(model)

param_optimizer = list(model.named_parameters())  # 模型参数名字列表
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

NUM_EPOCHS = 3
# optimizer = BertAdam(optimizer_grouped_parameters,
#                      lr=2e-5,
#                      warmup=0.05,
#                      t_total=len(train_loader) * NUM_EPOCHS
#                     )

# optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)   # 简单起见，可用这一行代码完事