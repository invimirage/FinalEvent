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
import json
import logging
import torch.nn.utils.rnn as rnn
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
from models import *
from sklearn.metrics import precision_recall_fscore_support
import os


class TextScorer:
    def __init__(self, **kwargs):
        self.logger = kwargs["logger"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"
        self.logger.info("Device: %s" % self.device)
        self.tag = torch.tensor(kwargs["tag"])
        self.data_len = self.tag.shape[0]
        self.logger.info("Data length: %d" % self.data_len)
        self.model = kwargs["model"]
        self.best_f1 = kwargs["f1"]
        # assert kwargs["mode"] in ["local", "api"]
        # self.mode = kwargs['mode']
        # if kwargs["mode"] == "local":
        #     bert_path = "../Models/Bert/"
        #     # Do embedding
        #     if kwargs['do_embed']:
        #         self.embed = self.bert_embedding(
        #             bert_path, self.data
        #         )
        #         self.data_input = self.embed
        #     else:
        #         self.data_input = self.data
        #     self.model = kwargs['model']
        #     # self.model = DoubleNet(input_length=1024 + kwargs['extra_feat_len'], hidden_length=32, drop_out_rate=0.4).to(
        #     #         self.device
        #     # self.model = SeparatedLSTM(input_length=1024 + kwargs['extra_feat_len'], hidden_length=128, layer_number=2, linear_hidden_length=32).to(self.device)
        # else:
        #     if kwargs['do_embed']:
        #         # 如果直接使用tencent API做向量化
        #         self.data_input_tensor = torch.from_numpy(
        #             self.tencent_embedding(self.data)
        #         ).to(torch.float32)
        #     else:
        #         self.data_input_tensor = torch.from_numpy(self.data).to(torch.float32)
        #     self.model = deal_embed(bert_out_length=768, hidden_length=4)

        self.model.to(self.device)

    def bert_embedding(
        self,
        bert_path,
        text_data,
        cls=False,
        neighbor=0,
        only_center=True,
        max_concat_len=100,
    ):
        tokenizer = BertTokenizer(
            vocab_file=os.path.join(bert_path, "vocab.txt")
        )  # 初始化分词器

        # 如果第一段文本有10小段，则记录为[0, 10]，该list元素数量比tag多1，tag[i]对应文本separate_points[i]:separates[i+1]
        separated_texts = []
        centers = []
        separated_points = [0]
        max_text_length = 512
        for slices in text_data:
            for i, text in enumerate(slices):
                target_text = text
                center_text = [1 for _ in range(len(text))]
                for j in range(1, neighbor + 1):
                    if (
                        i - j >= 0
                        and len(target_text) + len(slices[i - j]) <= max_concat_len
                    ):
                        target_text = slices[i - j] + target_text
                        center_text = [
                            0 for _ in range(len(slices[i - j]))
                        ] + center_text
                    if (
                        i + j < len(slices)
                        and len(target_text) + len(slices[i + j]) <= max_concat_len
                    ):
                        target_text = target_text + slices[i + j]
                        center_text = center_text + [
                            0 for _ in range(len(slices[i + j]))
                        ]
                centers.append(center_text)
                separated_texts.append(target_text)
            separated_points.append(len(separated_texts))

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

            center_text_masks = centers[sta:end]
            max_len = max([len(single) for single in tokens])  # 最大的句子长度

            for j in range(len(tokens)):
                padding = [0] * (max_len - len(tokens[j]))
                center_text_masks[j] += padding
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
            center_text_masks = torch.tensor(input_masks).to(self.device)

            with torch.no_grad():
                output = bert_model(
                    tokens_tensor,
                    token_type_ids=segments_tensors,
                    attention_mask=input_masks_tensors,
                )
            last_encode = output[0]
            if only_center:
                output_mask = center_text_masks.unsqueeze(-1).repeat(
                    1, 1, last_encode.shape[-1]
                )
            else:
                output_mask = input_masks_tensors.unsqueeze(-1).repeat(
                    1, 1, last_encode.shape[-1]
            )
            masked_output = last_encode * output_mask
            self.logger.debug(masked_output.shape)
            if cls:
                if only_center:
                    self.logger.fatal("Using centering and cls is wrong")
                    exit(0)
                pooled_output = masked_output[:, 0]
            else:
                pooled_output = torch.mean(masked_output, dim=1)
            self.logger.debug(pooled_output.shape)
            embed = pooled_output.cpu().detach().tolist()
            self.logger.debug(len(embed))
            embeds.extend(embed)
            torch.cuda.empty_cache()

        # 按照文本分割的embedding
        embeds_per_text = []
        for i in range(len(separated_points) - 1):
            embeds_per_text.append(
                embeds[separated_points[i] : separated_points[i + 1]]
            )
        return embeds_per_text

    def run_model(self, mode="train", **kwargs):
        if self.model.name == "BertWithCNN":
            self.run_model_with_bert(mode, kwargs)
        elif self.model.name == "SeparatedLSTM":
            self.run_model_separated_lstm(mode, kwargs)
        elif self.model.name == "BiLSTMWithAttention":
            self.run_model_separated_lstm(mode, kwargs)
        # # 文本数据是分段的，需要构建模型输入数据，即input和seps
        # def build_model_input(text_data, extra_data, indexes):
        #     input = []
        #     seps = []
        #     for data_section_id in indexes:
        #         data_section = text_data[data_section_id]
        #         extra_feat = extra_data[data_section_id]
        #         section_length = len(data_section)
        #         seps.append(len(input))
        #         for i, single_data in enumerate(data_section):
        #             # 加入分段id
        #             input.append(list(single_data) + list(extra_feat) + [(i + 1) / section_length])
        #     seps.append(len(input))
        #     return torch.tensor(input, dtype=torch.float32).to(self.device), seps
        #
        #
        # self.logger.info("Running model, %s" % mode)
        #
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # extra_feats = kwargs['extra_features']
        # text_embed_data = self.data_input
        # tags = self.tag.to(torch.int64)
        # self.logger.debug('Training data length: %d\n Tag data length: %d' % (len(text_embed_data), self.data_len))
        # assert mode in ["train", "predict"]
        # if mode == "train":
        #     indexes = np.arange(self.data_len)
        #     np.random.shuffle(indexes)
        #     train_len = round(self.data_len * trainning_size)
        #     train_inds = indexes[:train_len]
        #     self.logger.info('Training data length: %d' % len(train_inds))
        #     test_inds = indexes[train_len:]
        #     # 生成的训练、测试数据供测试使用
        #     # 取训练集的1/10
        #     train_inds_select = train_inds[::4]
        #     training_sample_input, training_sample_seps = build_model_input(text_embed_data, extra_feats, train_inds_select)
        #     training_sample_tags = tags[train_inds_select]
        #
        #     testing_input, testing_seps = build_model_input(text_embed_data, extra_feats, test_inds)
        #     testing_tags = tags[test_inds]
        #
        #     n_batch = math.ceil(len(train_inds) / batch_size)
        #     self.logger.debug("Batch number: %d" % n_batch)
        #     best_micro_f1 = 0
        #     best_epoch = 0
        #     for epoch in range(num_epoch):
        #
        #         # if epoch % 10 == 0:
        #         #     self.logger.info('Epoch number: %d' % epoch)
        #
        #         if epoch % 1 == 0:
        #             cpc_pred_train, train_loss = self.model(training_sample_input, training_sample_tags.to(self.device), separates=training_sample_seps)
        #             cpc_pred_test, test_loss = self.model(testing_input, testing_tags.to(self.device), separates=testing_seps, detail=True)
        #             cpc_pred_worst = (
        #                 cpc_pred_test.cpu().detach().numpy()[:, 0].flatten()
        #             )
        #             top10 = np.array(cpc_pred_worst).argsort()[::-1][0:10]
        #             self.logger.info("Worst Top 10: {}".format(kwargs["ids"][top10]))
        #             for i in kwargs["text"][top10]:
        #                 self.logger.info(i)
        #             cpc_pred_best = (
        #                 cpc_pred_test.cpu().detach().numpy()[:, -1].flatten()
        #             )
        #             top10 = np.array(cpc_pred_best).argsort()[::-1][0:10]
        #             self.logger.info("Best Top 10: {}".format(kwargs["ids"][top10]))
        #             for i in kwargs["text"][top10]:
        #                 self.logger.info(i)
        #             cpc_pred_train = np.argmax(cpc_pred_train.cpu().detach(), axis=1)
        #             cpc_pred_test = np.argmax(cpc_pred_test.cpu().detach(), axis=1)
        #             train_tags_cpu = training_sample_tags.cpu()
        #             test_tags_cpu = testing_tags.cpu()
        #             self.logger.info("------------Epoch %d------------" % epoch)
        #             self.logger.info("Training set")
        #             self.logger.info("Loss: %.4lf" % train_loss.cpu().detach())
        #             p_class, r_class, f_class, _ = precision_recall_fscore_support(
        #                 cpc_pred_train, train_tags_cpu
        #             )
        #             self.logger.info(p_class)
        #             self.logger.info(r_class)
        #             self.logger.info(f_class)
        #             self.logger.info("Testing set")
        #             self.logger.info("Loss: %.4lf" % test_loss.cpu().detach())
        #             p_class, r_class, f_class, _ = precision_recall_fscore_support(
        #                 cpc_pred_test, test_tags_cpu
        #             )
        #             self.logger.info(p_class)
        #             self.logger.info(r_class)
        #             self.logger.info(f_class)
        #             f1_mean = np.mean(f_class)
        #             if f1_mean > best_micro_f1:
        #                 best_micro_f1 = f1_mean
        #                 best_epoch = epoch
        #             self.logger.info('Best Micro-F1: %.6lf, epoch %d' % (best_micro_f1, best_epoch))
        #
        #         for i in range(n_batch):
        #             start = i * batch_size
        #             # 别忘了这里用了sigmoid归一化
        #             data_inds = train_inds[start : start + batch_size]
        #             # data_inds = [9871, 21763, 30344, 3806, 7942]
        #             # print(data_inds)
        #             # print(separates)
        #             data, seps = build_model_input(text_embed_data, extra_feats, data_inds)
        #             _tags = tags[data_inds].to(self.device)
        #             cpc_pred, loss = self.model(
        #                 data, _tags, seps
        #             )  # text_hashCodes是一个32-dim文本特征
        #             optimizer.zero_grad()
        #             self.logger.debug(loss)
        #             loss.backward()
        #             for name, param in self.model.named_parameters():
        #                 self.logger.debug(param.grad)
        #             optimizer.step()
        #
        #         np.random.shuffle(train_inds)
        #
        #         for name, param in self.model.named_parameters():
        #             if name == "fcs.2.bias":
        #                 self.logger.debug(name, param)
        # else:
        #     pass
        #     # n_batch = self.data_len // batch_size
        #     # for i in range(n_batch):
        #     #     start = i * batch_size
        #     #     # 别忘了这里用了sigmoid归一化
        #     #     cpc_pred = self.model(
        #     #         input_data[:, start : start + batch_size]
        #     #     )  # text_hashCodes是一个32-dim文本特征
        #     #
        #     #     loss = F.mse_loss(cpc_pred, tags)

    # def get_results(self, text, tag, pred, filepath):
    #     df = pd.DataFrame(columns=('text', 'pred', 'tag'))
    #     df['text'] = list(text)
    #     df['pred'] = list(pred.cpu().detach().numpy()[:, 1])
    #     df['tag'] = list(tag.cpu().detach().numpy())
    #     df.to_csv(filepath, index=False)

    def run_model_separated_lstm(self, mode, kwargs):
        batch_size = kwargs["params"]["batch_size"]
        lr = kwargs["params"]["learning_rate"]
        training_size = kwargs["params"]["training_size"]
        num_epoch = kwargs["params"]["number_epochs"]
        random_batching = kwargs["params"]["random"] == 1

        # 文本数据是分段的，需要构建模型输入数据，即input和seps
        def feed_model(text_data, extra_data, tag_data, indexes, requires_grad=True):
            input = []
            lengths = []
            extra = []
            for data_section_id in indexes:
                data_section = text_data[data_section_id]
                extra_feat = extra_data[data_section_id]
                extra.append(extra_feat)
                input.append(torch.tensor(data_section, dtype=torch.float32))
                lengths.append(len(data_section))
            input_padded = rnn.pad_sequence(input, batch_first=True)
            self.logger.debug("padded {}".format(input_padded))
            _input_packed = rnn.pack_padded_sequence(
                input_padded, lengths=lengths, batch_first=True, enforce_sorted=False
            ).to(self.device)
            _tag_data = tag_data[indexes].to(self.device)
            _extra_data = torch.tensor(extra, dtype=torch.float32).to(self.device)
            if not requires_grad:
                with torch.no_grad():
                    pred, loss = self.model(_input_packed, _extra_data, _tag_data)
            else:
                pred, loss = self.model(_input_packed, _extra_data, _tag_data)
            return pred, loss

        self.logger.info("Running model, %s" % mode)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        extra_feats = kwargs["extra_features"]
        text_embed_data = self.data_input
        tags = self.tag.to(torch.int64)
        self.logger.debug(
            "Training data length: %d\n Tag data length: %d"
            % (len(text_embed_data), self.data_len)
        )
        assert mode in ["train", "predict"]
        if mode == "train":
            train_inds, test_inds = sep_train_test(self.data_len, tags, training_size)
            self.logger.info("Training data length: %d" % len(train_inds))
            # 生成的训练、测试数据供测试使用
            # 取训练集的1/10
            train_inds_sample = train_inds[:: int(training_size // (1 - training_size))]
            best_micro_f1 = 0
            best_epoch = 0
            for epoch in range(num_epoch):

                # if epoch % 10 == 0:
                #     self.logger.info('Epoch number: %d' % epoch)

                if epoch % 1 == 0:
                    train_batches = build_batch(train_inds_sample, batch_size, False)
                    test_batches = build_batch(test_inds, batch_size, False)
                    training_preds = []
                    training_losses = []
                    testing_preds = []
                    testing_losses = []
                    for batch_inds in train_batches:
                        pred, loss = feed_model(
                            text_embed_data,
                            extra_feats,
                            tags,
                            batch_inds,
                            requires_grad=False,
                        )
                        training_preds.append(pred)
                        training_losses.append(loss.unsqueeze(0))
                    pred_train = torch.cat(training_preds, dim=0)
                    train_loss = torch.mean(torch.cat(training_losses, dim=0))
                    for batch_inds in test_batches:
                        pred, loss = feed_model(
                            text_embed_data,
                            extra_feats,
                            tags,
                            batch_inds,
                            requires_grad=False,
                        )
                        testing_preds.append(pred)
                        testing_losses.append(loss.unsqueeze(0))
                    pred_test = torch.cat(testing_preds, dim=0)
                    test_loss = torch.mean(torch.cat(testing_losses, dim=0))

                    f1_mean = output_logs(
                        self,
                        epoch,
                        kwargs,
                        pred_train,
                        train_loss,
                        train_inds_sample,
                        pred_test,
                        test_loss,
                        test_inds,
                    )
                    if f1_mean > best_micro_f1:
                        best_micro_f1 = f1_mean
                        best_epoch = epoch
                        if f1_mean > self.best_f1:
                            self.best_f1 = f1_mean
                            save_the_best(
                                self.model,
                                f1_mean,
                                kwargs["ids"][test_inds],
                                tags[test_inds],
                                pred_test,
                                self.logger.name,
                            )

                    self.logger.info(
                        "Best Micro-F1: %.6lf, epoch %d" % (best_micro_f1, best_epoch)
                    )
                    if epoch - best_epoch > 10:
                        break

                train_batches = build_batch(
                    train_inds, batch_size, random_batching, tags
                )
                for batch_inds in train_batches:
                    _, loss = feed_model(
                        text_embed_data, extra_feats, tags, batch_inds
                    )  # text_hashCodes是一个32-dim文本特征
                    optimizer.zero_grad()
                    self.logger.debug(loss)
                    loss.backward()
                    for name, param in self.model.named_parameters():
                        self.logger.debug(param.grad)
                    optimizer.step()
                #
                # for name, param in self.model.named_parameters():
                #     if name == "fcs.2.bias":
                #         self.logger.debug(name, param)
        else:
            pass
            # n_batch = self.data_len // batch_size
            # for i in range(n_batch):

    def run_model_with_bert(self, mode, kwargs):
        batch_size = kwargs["params"]["batch_size"]
        lr = kwargs["params"]["learning_rate"]
        training_size = kwargs["params"]["training_size"]
        num_epoch = kwargs["params"]["number_epochs"]
        bert_path = config.bert_path
        random_batching = kwargs["params"]["random"] == 1

        # 文本数据是分段的，需要构建模型输入数据，即input和seps
        def build_bert_input(text_data, max_text_length=150):
            tokens, segments, input_masks = [], [], []
            tokenizer = BertTokenizer(
                vocab_file=os.path.join(bert_path, "vocab.txt")
            )  # 初始化分词器
            for text in text_data:
                indexed_tokens = tokenizer.encode(text)  # 索引列表
                if len(indexed_tokens) > max_text_length:
                    indexed_tokens = (
                        indexed_tokens[: max_text_length // 2]
                        + indexed_tokens[-max_text_length // 2 :]
                    )
                tokens.append(indexed_tokens)
                segments.append([0] * len(indexed_tokens))
                input_masks.append([1] * len(indexed_tokens))
            for j in range(len(tokens)):
                padding = [0] * (max_text_length - len(tokens[j]))
                tokens[j] += padding
                segments[j] += padding
                input_masks[j] += padding
            return tokens, segments, input_masks

        def feed_model(bert_input, extra_data, tag_data, indexes, train=True):
            input = [[], [], []]
            extra = []
            for data_section_id in indexes:
                for i in range(3):
                    input[i].append(bert_input[i][data_section_id])
                extra.append(extra_data[data_section_id])
            _tag_data = tag_data[indexes].to(self.device)
            _extra_data = torch.tensor(extra, dtype=torch.float32).to(self.device)
            input = [torch.tensor(data, device=self.device) for data in input]
            if train:
                pred, loss = self.model(input, _extra_data, _tag_data)
                return pred, loss
            else:
                # self.model.eval()
                with torch.no_grad():
                    pred, loss = self.model(input, _extra_data, _tag_data)
                    loss = loss.cpu().detach()
                    pred = pred.cpu().detach()
                # self.model.train()
                return pred, loss

        self.logger.info("Running model, %s" % mode)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        extra_feats = kwargs["extra_features"]
        text_data = kwargs["text"]
        bert_input = build_bert_input(text_data)
        tags = self.tag.to(torch.int64)
        self.logger.debug(
            "Training data length: %d\n Tag data length: %d"
            % (len(text_data), self.data_len)
        )
        assert mode in ["train", "predict"]
        if mode == "train":
            train_inds, test_inds = sep_train_test(self.data_len, tags, training_size)
            self.logger.info("Training data length: %d" % len(train_inds))
            # 生成的训练、测试数据供测试使用
            # 取训练集的1/10
            train_inds_sample = train_inds[:: int(training_size // (1 - training_size))]
            best_micro_f1 = 0
            best_epoch = 0
            for epoch in range(num_epoch):
                if epoch % 1 == 0:
                    train_batches = build_batch(train_inds_sample, batch_size, False)
                    test_batches = build_batch(test_inds, batch_size, False)
                    training_preds = []
                    training_losses = []
                    testing_preds = []
                    testing_losses = []
                    for batch_inds in train_batches:
                        pred, loss = feed_model(
                            bert_input,
                            extra_feats,
                            tags,
                            batch_inds,
                            False,
                        )
                        training_preds.append(pred)
                        training_losses.append(loss.unsqueeze(0))
                    pred_train = torch.cat(training_preds, dim=0)
                    train_loss = torch.mean(torch.cat(training_losses, dim=0))
                    for batch_inds in test_batches:
                        pred, loss = feed_model(
                            bert_input, extra_feats, tags, batch_inds, False
                        )
                        testing_preds.append(pred)
                        testing_losses.append(loss.unsqueeze(0))
                    pred_test = torch.cat(testing_preds, dim=0)
                    test_loss = torch.mean(torch.cat(testing_losses, dim=0))
                    f1_mean = output_logs(
                        self,
                        epoch,
                        kwargs,
                        pred_train,
                        train_loss,
                        train_inds_sample,
                        pred_test,
                        test_loss,
                        test_inds,
                    )
                    if f1_mean > best_micro_f1:
                        best_micro_f1 = f1_mean
                        best_epoch = epoch
                        if f1_mean > self.best_f1:
                            self.best_f1 = f1_mean
                            save_the_best(
                                self.model,
                                f1_mean,
                                kwargs["ids"][test_inds],
                                tags[test_inds],
                                pred_test,
                                self.logger.name,
                            )
                    self.logger.info(
                        "Best Micro-F1: %.6lf, epoch %d" % (best_micro_f1, best_epoch)
                    )
                    if epoch - best_epoch > 10:
                        break

                train_batches = build_batch(
                    train_inds, batch_size, random_batching, tags
                )
                for batch_inds in train_batches:
                    _, loss = feed_model(
                        bert_input, extra_feats, tags, batch_inds
                    )  # text_hashCodes是一个32-dim文本特征
                    optimizer.zero_grad()
                    self.logger.debug(loss)
                    loss.backward()
                    for name, param in self.model.named_parameters():
                        self.logger.debug(param.grad)
                    optimizer.step()
        else:
            pass

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

    def set_data(self, input_data):
        assert len(input_data) == self.tag.shape[0]
        self.data_input = input_data


def gen_correct_data(text_data, embeds):
    separated_points = [0]
    total_len = 0
    for slices in text_data:
        total_len += len(slices)
        separated_points.append(total_len)
    embeds_flat = sum(embeds, [])
    embeds_per_text = []
    for i in range(len(separated_points) - 1):
        embeds_per_text.append(
            embeds_flat[separated_points[i] : separated_points[i + 1]]
        )
    return embeds_per_text


def main(model, logger, kwargs) -> None:
    method_name = logger.name
    embed_type = kwargs["embed_type"]
    requires_embedding = model.requires_embed
    run_params = model.hyperparams["common"]
    assert embed_type in ["api", "local"]
    # embed_data = np.load('../Data/vector_embed.npz')
    # dropped_data = embed_data['dropped_ind']
    print(config.raw_data_file)
    tag_col = kwargs.get("tag_col", "tag")
    print(tag_col)
    data = pd.read_csv(
        config.raw_data_file,
        # usecols=["id", "like", "clk", "separated_text", "advid", tag_col],
    )
    if kwargs.get("test_mode", False):
        sep = len(data) // 200
        data = data[::sep]
        data.reset_index(inplace=True)
    # tag_data = np.array(data['bctr_tag'])
    tag_data = np.array(data[tag_col])
    # advid_avg_bctr = pd.read_csv('Data/kuaishou_data_0426/bctr_avg.csv')
    # advid_avg_bctr.set_index('advid', inplace=True)
    # advid_avg_bctr = advid_avg_bctr.to_dict(orient='index')
    # # data.drop(index=data.index[dropped_data]
    # imp = np.array(data["clk"])
    # beh = np.array(data["bclk"])
    key = np.array(data["key"])
    # advids = data['advid']
    #
    #
    # datalen = len(data)
    # removed_indexes = []
    # tag_data = []
    # for i in range(datalen):
    #     advid = advids[i]
    #     if advid_avg_bctr[advid]['count'] < 10:
    #         removed_indexes.append(i)
    #         continue
    #     else:
    #         mean, std = advid_avg_bctr[advid]['bctr'], advid_avg_bctr[advid]['std']
    #         tag_data.append((imp[i] / (beh[i] + 1e-10) - mean) / std)
    #     # if imp[i] < config.threshold:
    #     #     beh[i] = 0
    #     # # 避免分母为0
    #     # if imp[i] == 0:
    #     #     imp[i] = 1
    # # tag_data = beh / imp
    # tag_data = np.array(tag_data)

    # # print(tag_data.mean(), tag_data.max(), tag_data.min())
    # # plt.hist(tag_data, 10, range=(0, 0.2), facecolor="blue", edgecolor="black", alpha=0.7)
    # # plt.show()
    # TextScorer(data=text_data, tag=tag_data, mode='api', lr=1e-3)

    # vector_data = embed_data['embed']
    # tag_data = np.delete(tag_data, embed_data['dropped_ind'])

    text_data = data["separated_text"].apply(lambda text: json.loads(text))
    # binned_data, cut_points = bin_tags(tag_data, config.bin_number)
    best_f1 = load_model(file_name=logger.name)

    scorer = TextScorer(
        tag=tag_data, mode=embed_type, logger=logger, model=model, f1=best_f1
    )

    if requires_embedding:
        embed_file_path = kwargs.get("embed_file", config.embed_data_file + f"_{method_name}.npy")
        if kwargs["force_embed"]:
            print("Re embedding")
            embed = scorer.bert_embedding(
                config.bert_path,
                text_data,
                kwargs.get("use_cls", False),
                kwargs.get("neighbor", 0),
                kwargs.get("only_center", False),
                kwargs.get("max_len", 100),
            )
            embed_cache = np.array(embed, dtype=object)
            del embed
            np.save(embed_file_path, embed_cache)
            embed_data = embed_cache.tolist()
            del embed_cache
            scorer.set_data(embed_data)
        else:
            try:
                embed_data = np.load(embed_file_path, allow_pickle=True).tolist()
                embed_data = gen_correct_data(text_data, embed_data)
                # embed_data = [embed_data[i] for i in range(len(embed_data)) if i not in removed_indexes]
            except:
                print("Text embedding not found! Now doing embedding")
                embed = scorer.bert_embedding(
                    config.bert_path,
                    text_data,
                    kwargs.get("use_cls", False),
                    kwargs.get("neighbor", 0),
                    kwargs.get("only_center", False),
                    kwargs.get("max_len", 100),
                )
                embed_cache = np.array(embed, dtype=object)
                np.save(embed_file_path, embed_cache)
                embed_data = embed_cache.tolist()
            finally:
                scorer.set_data(embed_data)

    extra_features = parse_extra_features(data)

    # 我觉得过一段时间就很难看懂了，不过还是很帅
    # process_mark = np.concatenate(
    #     [np.arange(1, 1 + sep_points[i + 1] - sep_points[i]) / (sep_points[i + 1] - sep_points[i])
    #      for i in range(len(sep_points) - 1)])
    # process_mark = process_mark.reshape([len(process_mark), -1])
    # extra_features = np.concatenate([advid_onehot, process_mark], axis=1)

    # scorer.run_model(num_epoch=10000, trainning_size=0.8, extra_features=advid_onehot, batch_size=500, lr=1e-3, ids=id, text=text_data)
    # scorer.run_model_with_bert(bert_path='../Models/Bert/', num_epoch=10000, trainning_size=0.8,
    #                            extra_features=advid_onehot, batch_size=50, lr=1e-3, ids=id, text=text_data)

    scorer.run_model(
        mode="train",
        extra_features=extra_features,
        ids=key,
        text=text_data,
        params=run_params,
    )


if __name__ == "__main__":
    # data_source raw embed
    # embed_type local api
    with open(config.parameter_file) as f:
        params = json.load(f)
    logger = init_logger(
        log_level=logging.INFO,
        name="BertWithCNN_with_advid",
        write_to_file=True,
        clean_up=True,
    )
    adjust_hyperparams(
        logger,
        params,
        sample_number=10,
        model_name="BertWithCNN",
        run_model=main,
        embed_type="local",
    )
    # if model_name == 'SeparatedLSTM':
    #     model = SeparatedLSTM(
    #         input_length=1024, extra_length=len(config.advids), hyperparams=params[model_name]
    #     )
    # elif model_name == 'BertWithCNN':
    #     model = BertWithCNN(bert_path=config.bert_path, extra_length=0, hyperparams=params['BertWithCNN'])
    # main(model=model, logger=logger, kwargs={"embed_type": "local"})
    # slstm = SeparatedLSTM(input_length=1024, extra_length=0, hyperparams=params['SeparatedLSTM'])
    # bwc = BertWithCNN(bert_path=config.bert_path, extra_length=0, hyperparams=params['BertWithCNN'])
    # main(model=slstm, embed_type="local", log_level=logging.INFO)
    # 以后就不需要转换embed了
