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
from tqdm import tqdm
import math
# from torch.utils.data import *
from basic import DataParser
import pandas as pd
import json
import logging
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import (
    TencentCloudSDKException,
)
from tencentcloud.nlp.v20190408 import nlp_client, models
from utils import *
import config
from matplotlib import pyplot as plt
from model import *
from sklearn.metrics import precision_recall_fscore_support


class TextScorer:
    def __init__(self, **kwargs):
        logging.basicConfig(
            format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S"
        )
        self.logger = logging.getLogger("log")
        self.logger.setLevel(kwargs["log_level"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"
        self.logger.info("Device: %s" % self.device)
        self.data = kwargs["data"]
        self.tag = torch.tensor(kwargs["tag"])
        self.data_len = self.tag.shape[0]
        self.logger.info("Data length: %d" % self.data_len)
        assert kwargs["mode"] in ["local", "api"]
        # self.mode = kwargs['mode']
        self.batch_size = kwargs["batch_size"]
        if kwargs["mode"] == "local":
            bert_path = "../Models/Bert/"
            # Do embedding
            if isinstance(self.data[0], list):
                self.embed, self.separate_points = self.bert_embedding(
                    bert_path, self.data
                )
                self.data_input_tensor = torch.tensor(self.embed, dtype=torch.float32)
            else:
                self.data_input_tensor = torch.from_numpy(self.data).to(torch.float32)
            self.model = DoubleNet(input_length=1024 + kwargs['extra_feat_len'], hidden_length=32, drop_out_rate=0.4).to(
                self.device
            )
        else:
            if isinstance(self.data[0], str):
                # 如果直接使用tencent API做向量化
                self.data_input_tensor = torch.from_numpy(
                    self.tencent_embedding(self.data)
                ).to(torch.float32)
            else:
                self.data_input_tensor = torch.from_numpy(self.data).to(torch.float32)
            self.model = deal_embed(bert_out_length=768, hidden_length=4)

        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=kwargs["lr"])

    def bert_embedding(self, bert_path, text_data):
        tokenizer = BertTokenizer(vocab_file=bert_path + "vocab.txt")  # 初始化分词器

        # 如果第一段文本有10小段，则记录为[0, 10]，该list元素数量比tag多1，tag[i]对应文本separate_points[i]:separates[i+1]
        separate_points = []
        separated_texts = []
        max_text_length = 512
        for slices in text_data:
            separate_points.append(len(separated_texts))
            for text in slices:
                separated_texts.append(text)
        separate_points.append(len(separated_texts))

        max_len = max([len(single) for single in separated_texts])  # 最大的句子长度
        self.logger.info("data_size: %d" % len(text_data))
        self.logger.info("max_seq_len: %d" % max_len)
        self.logger.info(
            "avg_seq_len: %d " % np.mean([len(single) for single in separated_texts])
        )

        bert_model = BertModel.from_pretrained(bert_path).to(self.device)
        bert_model.eval()
        batch_size = config.bert_batch_size
        n_batch = math.ceil(len(separated_texts) / batch_size)
        embeds = []
        for i in range(n_batch):
            if i % 100 == 0:
                self.logger.info("Embedding, %d / %d" % (i, n_batch))
            # if i != 0 and i % max_save_size == 0:
            sta = i * batch_size
            end = (i + 1) * batch_size
            tokens, segments, input_masks = [], [], []
            for text in separated_texts[sta:end]:
                indexed_tokens = tokenizer.encode(text)  # 索引列表
                if len(indexed_tokens) > max_text_length:
                    indexed_tokens = indexed_tokens[:max_text_length]
                tokens.append(indexed_tokens)
                segments.append([0] * len(indexed_tokens))
                input_masks.append([1] * len(indexed_tokens))

            max_len = max([len(single) for single in tokens])  # 最大的句子长度

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

            output = bert_model(
                tokens_tensor,
                token_type_ids=segments_tensors,
                attention_mask=input_masks_tensors,
            )
            last_encode = output[0]
            output_mask = (
                input_masks_tensors
                .unsqueeze(-1)
                .repeat(1, 1, last_encode.shape[-1])
            )
            masked_output = last_encode * output_mask
            self.logger.debug(masked_output.shape)
            pooled_output = torch.mean(masked_output, dim=1)
            self.logger.debug(pooled_output.shape)
            embed = pooled_output.cpu().detach().tolist()
            self.logger.debug(len(embed))
            embeds.extend(embed)
            torch.cuda.empty_cache()
        return embeds, separate_points

    def run_model(self, mode="train", trainning_size=0.8, num_epoch=10, **kwargs):
        self.logger.info("Running model, %s" % mode)
        extra_feats = torch.from_numpy(kwargs['extra_features']).to(torch.float32)
        input_data = torch.cat((self.data_input_tensor[:352639], extra_feats), dim=1)
        tags = self.tag.to(torch.int64)
        assert mode in ["train", "predict"]
        if mode == "train":
            indexes = np.arange(self.data_len)
            np.random.shuffle(indexes)
            train_len = round(self.data_len * trainning_size)
            train_inds = indexes[:train_len]
            self.logger.info('Training data length: %d' % len(train_inds))
            test_inds = indexes[train_len:]
            self.separate_points = np.array(self.separate_points, dtype=int)
            # 生成的训练、测试数据供测试使用
            # 取训练集的1/10
            train_inds_select = train_inds[::10]
            map_to_training = self.separate_points[train_inds_select], self.separate_points[np.array(train_inds_select)+1]
            train_data = input_data[np.concatenate([np.arange(sta, end) for sta, end in zip(*map_to_training)])]
            train_tags = tags[train_inds_select]
            map_lens_train = [end - sta for sta, end in zip(*map_to_training)]
            separates_train = [0]
            for map_len in map_lens_train:
                separates_train.append(separates_train[-1] + map_len)

            map_to_testing = self.separate_points[test_inds], self.separate_points[np.array(test_inds)+1]
            test_data = input_data[np.concatenate([np.arange(sta, end) for sta, end in zip(*map_to_testing)])]
            test_tags = tags[test_inds]
            map_lens_test = [end - sta for sta, end in zip(*map_to_testing)]
            separates_test = [0]
            for map_len in map_lens_test:
                separates_test.append(separates_test[-1] + map_len)

            n_batch = math.ceil(len(train_inds) / self.batch_size)
            self.logger.debug("Batch number: %d" % n_batch)
            best_micro_f1 = 0
            best_epoch = 0
            for epoch in range(num_epoch):

                # if epoch % 10 == 0:
                #     self.logger.info('Epoch number: %d' % epoch)

                if epoch % 10 == 0:
                    cpc_pred_train, train_loss = self.model(train_data.to(self.device), train_tags.to(self.device), separates=separates_train)
                    cpc_pred_test, test_loss = self.model(test_data.to(self.device), test_tags.to(self.device), separates=separates_test)
                    cpc_pred_worst = (
                        cpc_pred_test.cpu().detach().numpy()[:, 0].flatten()
                    )
                    top10 = np.array(cpc_pred_worst).argsort()[::-1][0:10]
                    self.logger.info("Worst Top 10: {}".format(kwargs["ids"][top10]))
                    for i in kwargs["text"][top10]:
                        self.logger.info(i)
                    cpc_pred_best = (
                        cpc_pred_test.cpu().detach().numpy()[:, -1].flatten()
                    )
                    top10 = np.array(cpc_pred_best).argsort()[::-1][0:10]
                    self.logger.info("Best Top 10: {}".format(kwargs["ids"][top10]))
                    for i in kwargs["text"][top10]:
                        self.logger.info(i)
                    cpc_pred_train = np.argmax(cpc_pred_train.cpu().detach(), axis=1)
                    cpc_pred_test = np.argmax(cpc_pred_test.cpu().detach(), axis=1)
                    train_tags_cpu = train_tags.cpu()
                    test_tags_cpu = test_tags.cpu()
                    self.logger.info("------------Epoch %d------------" % epoch)
                    self.logger.info("Training set")
                    self.logger.info("Loss: %.4lf" % train_loss.cpu().detach())
                    p_class, r_class, f_class, _ = precision_recall_fscore_support(
                        cpc_pred_train, train_tags_cpu
                    )
                    self.logger.info(p_class)
                    self.logger.info(r_class)
                    self.logger.info(f_class)
                    self.logger.info("Testing set")
                    self.logger.info("Loss: %.4lf" % test_loss.cpu().detach())
                    p_class, r_class, f_class, _ = precision_recall_fscore_support(
                        cpc_pred_test, test_tags_cpu
                    )
                    self.logger.info(p_class)
                    self.logger.info(r_class)
                    self.logger.info(f_class)
                    f1_mean = np.mean(f_class)
                    if f1_mean > best_micro_f1:
                        best_micro_f1 = f1_mean
                        best_epoch = epoch
                    self.logger.info('Best Micro-F1: %.6lf, epoch %d' % (best_micro_f1, best_epoch))

                for i in range(n_batch):
                    start = i * self.batch_size
                    # 别忘了这里用了sigmoid归一化
                    data_inds = train_inds[start : start + self.batch_size]
                    # data_inds = [9871, 21763, 30344, 3806, 7942]
                    # print(data_inds)
                    map_to_data = self.separate_points[data_inds], self.separate_points[np.array(data_inds) + 1]
                    map_lens = [end - sta for sta, end in zip(*map_to_data)]
                    separates = [0]
                    for map_len in map_lens:
                        separates.append(separates[-1] + map_len)
                    # print(separates)
                    data = input_data[np.concatenate([np.arange(sta, end) for sta, end in zip(*map_to_data)])].to(self.device)
                    _tags = tags[data_inds].to(self.device)
                    cpc_pred, loss = self.model(
                        data, _tags, separates
                    )  # text_hashCodes是一个32-dim文本特征
                    self.optimizer.zero_grad()
                    self.logger.debug(loss)
                    loss.backward()
                    for name, param in self.model.named_parameters():
                        self.logger.debug(param.grad)
                    self.optimizer.step()

                np.random.shuffle(train_inds)

                for name, param in self.model.named_parameters():
                    if name == "fcs.2.bias":
                        self.logger.debug(name, param)
        else:
            n_batch = self.data_len // self.batch_size
            for i in range(n_batch):
                start = i * self.batch_size
                # 别忘了这里用了sigmoid归一化
                cpc_pred = self.model(
                    input_data[:, start : start + self.batch_size]
                )  # text_hashCodes是一个32-dim文本特征

                loss = F.mse_loss(cpc_pred, tags)

    def tencent_embedding(self, sentences):
        vector_embeds = []
        failed_inds = []
        for ind, sentence in enumerate(sentences):
            if ind % 100 == 0:
                self.logger.info("Now at {}, {} in total".format(ind, len(sentences)))
            try:
                cred = credential.Credential(
                    "AKIDlJdkbExRlwueDaqjZAaomVFlDSVOuqCL",
                    "iTefWR6XklmIfroVyQergHqAG9qIsvkO",
                )
                httpProfile = HttpProfile()
                httpProfile.endpoint = "nlp.tencentcloudapi.com"

                clientProfile = ClientProfile()
                clientProfile.httpProfile = httpProfile
                client = nlp_client.NlpClient(cred, "ap-guangzhou", clientProfile)

                req = models.SentenceEmbeddingRequest()
                params = {"Text": sentence}
                req.from_json_string(json.dumps(params))

                resp = client.SentenceEmbedding(req)
                vector = json.loads(resp.to_json_string())["Vector"]
                vector_embeds.append(vector)

            except TencentCloudSDKException as err:
                print(err)
                failed_inds.append(ind)
                # ——————输入处理——————
        np.savez(
            "vector_embed",
            embed=np.array(vector_embeds),
            dropped_ind=np.array(failed_inds),
        )
        return vector_embeds


def main(data_source: str, embed_type: str, log_level: str, data_path: str = "../Data/kuaishou_data_es.csv") -> None:
    assert data_source in ["raw", "embed"]
    assert embed_type in ["api", "local"]
    # embed_data = np.load('../Data/vector_embed.npz')
    # dropped_data = embed_data['dropped_ind']
    data = pd.read_csv(data_path)
    # data.drop(index=data.index[dropped_data]
    imp = np.array(data["clk"])
    beh = np.array(data["bclk"])
    id = np.array(data["id"])

    datalen = len(data)
    for i in range(datalen):
        if imp[i] < config.threshold:
            beh[i] = 0
        # 避免分母为0
        if imp[i] == 0:
            imp[i] = 1
    tag_data = beh / imp
    # # print(tag_data.mean(), tag_data.max(), tag_data.min())
    # # plt.hist(tag_data, 10, range=(0, 0.2), facecolor="blue", edgecolor="black", alpha=0.7)
    # # plt.show()
    # TextScorer(data=text_data, tag=tag_data, mode='api', lr=1e-3)

    # vector_data = embed_data['embed']
    # tag_data = np.delete(tag_data, embed_data['dropped_ind'])

    binned_data, cut_points = bin_tags(tag_data, config.bin_number)
    text_data = data["separated_text"].apply(lambda text: json.loads(text))

    if data_source == "raw":
        scorer = TextScorer(
            data=text_data,
            tag=binned_data,
            mode=embed_type,
            lr=1e-2,
            batch_size=1000,
            log_level=log_level,
            extra_feat_len=1+len(config.advids)
        )
        embed_data = scorer.embed
        sep_points = scorer.separate_points
        np.savez(
            "../Data/vector_embed",
            embed=np.array(embed_data),
            sep_points=np.array(sep_points),
        )

    else:
        try:
            embed_datafile = np.load("../Data/vector_embed.npz")

            # 以后就不需要转换了应该
            embed_data = np.reshape(embed_datafile['embed'], (-1, embed_datafile['embed'].shape[-1]))
            sep_points = embed_datafile['sep_points'][:-1]
            binned_data = binned_data[:-1]
            # print(binned_data.shape, sep_points.shape, embed_data.shape)
            # print(sep_points[-1])
            # print(sep_points[-5:], embed_data.shape)
            embed_data = np.array(embed_data, dtype=float)
        except:
            print("Text embedding not found!")
            exit(0)
        scorer = TextScorer(
            data=embed_data,
            tag=binned_data,
            mode=embed_type,
            lr=1e-3,
            batch_size=1000,
            log_level=log_level,
            extra_feat_len=1 + len(config.advids)
        )
        scorer.separate_points = sep_points
    scorer.logger.info(cut_points)
    advid_onehot = []
    one_hot_len = len(config.advids)
    for i in range(scorer.data_len):
        advid = data['advid'][i]
        sep_len = sep_points[i+1] - sep_points[i]
        try:
            idx = config.advids.index(str(advid))
            one_hot = np.eye(one_hot_len, dtype=int)[idx]
        except ValueError:
            one_hot = np.zeros(one_hot_len, dtype=int)
        for j in range(sep_len):
            advid_onehot.append(one_hot)

    # 我觉得过一段时间就很难看懂了，不过还是很帅
    process_mark = np.concatenate(
        [np.arange(1, 1 + sep_points[i + 1] - sep_points[i]) / (sep_points[i + 1] - sep_points[i])
         for i in range(len(sep_points) - 1)])
    process_mark = process_mark.reshape([len(process_mark), -1])
    extra_features = np.concatenate([advid_onehot, process_mark], axis=1)
    scorer.run_model(num_epoch=10000, trainning_size=0.8, extra_features=extra_features, ids=id, text=text_data)


if __name__ == "__main__":
    # data_source raw embed
    # embed_type local api
    main(data_source="embed", embed_type="local", log_level=logging.INFO, data_path='../Data/kuaishou_data_0421.csv')
    # 以后就不需要转换embed了

