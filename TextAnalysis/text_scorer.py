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
from tqdm import tqdm
# from torch.utils.data import *
from basic import DataParser
import pandas as pd
import json
import logging
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.nlp.v20190408 import nlp_client, models
from utils import *
import config
from matplotlib import pyplot as plt
from model import *
from sklearn.metrics import precision_recall_fscore_support

class TextScorer:
    def __init__(self, **kwargs):
        logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
        self.logger = logging.getLogger('log')
        self.logger.setLevel(kwargs['log_level'])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info('Device: %s' % self.device)
        self.data = kwargs['data']
        self.tag = torch.tensor(kwargs['tag'])
        self.data_len = len(self.data)
        self.logger.info('Data length: %d' % self.data_len)
        assert kwargs['mode'] in ['local', 'api']
        self.batch_size = kwargs['batch_size']
        if kwargs['mode'] == 'local':
            bert_path = "Bert/"
            self.data_input_tensor = self.build_bert_input(bert_path, self.data)
            self.model = TextNet(hidden_length=128, bert_path=bert_path).to(self.device)

        else:
            if type(self.data[0]) == 'str':
                # 如果直接使用tencent API做向量化
                self.data_input_tensor = torch.from_numpy(self.sentence_embedding(self.data)).to(torch.float32)
            else:
                self.data_input_tensor = torch.from_numpy(self.data).to(torch.float32)
            self.model = deal_embed(bert_out_length=768, hidden_length=4)

        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=kwargs['lr'])


    def build_bert_input(self, bert_path, text_data):
        tokenizer = BertTokenizer(vocab_file=bert_path + "vocab.txt")  # 初始化分词器
        tokens, segments, input_masks = [], [], []
        for text in text_data:
            indexed_tokens = tokenizer.encode(text)  # 索引列表
            tokens.append(indexed_tokens)
            segments.append([0] * len(indexed_tokens))
            input_masks.append([1] * len(indexed_tokens))

        max_len = max([len(single) for single in tokens])  # 最大的句子长度
        self.logger.debug('batch_size: %d' % len(text_data))
        self.logger.debug('max_seq_len: %d' % max_len)
        self.logger.debug('avg_seq_len: %d ' % np.mean([len(single) for single in tokens]))
        for j in range(len(tokens)):
            padding = [0] * (max_len - len(tokens[j]))
            tokens[j] += padding
            segments[j] += padding
            input_masks[j] += padding
        # segments列表全0，因为只有一个句子1，没有句子2
        # input_masks列表1的部分代表句子单词，而后面0的部分代表paddig，只是用于保持输入整齐，没有实际意义。
        # 相当于告诉BertModel不要利用后面0的部分

        # 转换成PyTorch tensors
        tokens_tensor = torch.tensor(tokens).to(self.device)
        segments_tensors = torch.tensor(segments).to(self.device)
        input_masks_tensors = torch.tensor(input_masks).to(self.device)
        return tokens_tensor, segments_tensors, input_masks_tensors

    def run_model(self, mode='train', trainning_size=0.8, num_epoch=10):
        self.logger.info('Running model, %s' % mode)
        input_data = self.data_input_tensor.to(self.device)
        tags = self.tag.to(device=self.device, dtype=torch.int64)
        assert mode in ['train', 'predict']
        if mode == 'train':
            indexes = np.arange(self.data_len)
            np.random.shuffle(indexes)
            train_len = round(self.data_len * trainning_size)
            train_inds = indexes[:train_len]
            test_inds = indexes[train_len:]
            train_data = input_data[train_inds]
            train_tags = tags[train_inds]
            test_data = input_data[test_inds]
            test_tags = tags[test_inds]
            n_batch = len(train_data) // self.batch_size
            self.logger.debug('Batch number: %d' % n_batch)
            for epoch in range(num_epoch):

                if epoch % 100 == 0:
                    cpc_pred_train, _ = self.model(train_data)
                    cpc_pred_test, _ = self.model(test_data)
                    cpc_pred_train = np.argmax(cpc_pred_train.cpu().detach(), axis=1)
                    cpc_pred_test = np.argmax(cpc_pred_test.cpu().detach(), axis=1)
                    train_tags_cpu = train_tags.cpu()
                    test_tags_cpu = test_tags.cpu()
                    self.logger.info('------------Epoch %d------------' % epoch)
                    self.logger.info('Training set')
                    p_class, r_class, f_class, _ = precision_recall_fscore_support(cpc_pred_train, train_tags_cpu)
                    self.logger.info(p_class)
                    self.logger.info(r_class)
                    self.logger.info(f_class)
                    self.logger.info('Testing set')
                    p_class, r_class, f_class, _ = precision_recall_fscore_support(cpc_pred_test, test_tags_cpu)
                    self.logger.info(p_class)
                    self.logger.info(r_class)
                    self.logger.info(f_class)

                for i in range(n_batch):
                    start = i * self.batch_size
                    # 别忘了这里用了sigmoid归一化
                    cpc_pred, loss = self.model(train_data[start: start + self.batch_size], train_tags[start: start + self.batch_size])  # text_hashCodes是一个32-dim文本特征
                    self.optimizer.zero_grad()
                    self.logger.debug(loss)
                    loss.backward()
                    for name, param in self.model.named_parameters():
                        self.logger.debug(param.grad)
                    self.optimizer.step()

                for name, param in self.model.named_parameters():
                    if name == 'fcs.2.bias':
                        self.logger.debug(name, param)
        else:
            n_batch = self.data_len // self.batch_size
            for i in range(n_batch):
                start = i * self.batch_size
                # 别忘了这里用了sigmoid归一化
                cpc_pred = self.model(input_data[:, start: start + self.batch_size])  # text_hashCodes是一个32-dim文本特征

                loss = F.mse_loss(cpc_pred, tags)


    def sentence_embedding(self, sentences):
        vector_embeds = []
        failed_inds = []
        for ind, sentence in enumerate(sentences):
            if ind % 100 == 0:
                self.logger.info('Now at {}, {} in total'.format(ind, len(sentences)))
            try:
                cred = credential.Credential("AKIDlJdkbExRlwueDaqjZAaomVFlDSVOuqCL", "iTefWR6XklmIfroVyQergHqAG9qIsvkO")
                httpProfile = HttpProfile()
                httpProfile.endpoint = "nlp.tencentcloudapi.com"

                clientProfile = ClientProfile()
                clientProfile.httpProfile = httpProfile
                client = nlp_client.NlpClient(cred, "ap-guangzhou", clientProfile)

                req = models.SentenceEmbeddingRequest()
                params = {
                    "Text": sentence
                }
                req.from_json_string(json.dumps(params))

                resp = client.SentenceEmbedding(req)
                vector = json.loads(resp.to_json_string())["Vector"]
                vector_embeds.append(vector)

            except TencentCloudSDKException as err:
                print(err)
                failed_inds.append(ind)
                # ——————输入处理——————
        np.savez('vector_embed', embed=np.array(vector_embeds), dropped_ind=np.array(failed_inds))
        return vector_embeds

if __name__ == '__main__':
    data_path = r'C:\Users\zrf19\PycharmProjects\pythonProject\FinalEvent\RawDataURL\kuaishou_data_es.csv'
    data = pd.read_csv(data_path)
    text_data = list(data['full_texts'])
    imp = np.array(data['clk'])
    beh = np.array(data['bclk'])
    id = np.array(data['id'])

    tag_data = beh / imp
    # # print(tag_data.mean(), tag_data.max(), tag_data.min())
    # # plt.hist(tag_data, 10, range=(0, 0.2), facecolor="blue", edgecolor="black", alpha=0.7)
    # # plt.show()
    # TextScorer(data=text_data, tag=tag_data, mode='api', lr=1e-3)

    embed_data = np.load('vector_embed.npz')
    vector_data = embed_data['embed']
    tag_data = np.delete(tag_data, embed_data['dropped_ind'])

    binned_data, cut_points = bin_tags(tag_data, config.bin_number)
    print(cut_points)
    scorer = TextScorer(data=vector_data, tag=binned_data, mode='api', lr=1e-2, batch_size=2000, log_level=logging.INFO)
    scorer.run_model(num_epoch=10000, trainning_size=0.8)