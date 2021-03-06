#!/usr/bin/env python
# encoding: utf-8
'''
@author: zhangruifeng
@contact: zrf1999@pku.edu.cn
@file: long_text_scorer.py
@time: 2021/4/28 13:56
@github: local 16351726fa15c85f565b7d5fecdf320ea67a72ef
'''

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
from models import *
from sklearn.metrics import precision_recall_fscore_support


class LongTextScorer:
    def __init__(self, data_file_path, **kwargs):
        logging.basicConfig(
            format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S"
        )
        self.logger = logging.getLogger("log")
        self.logger.setLevel(kwargs["log_level"])
        with open('Hyperparameters.json') as f:
            self.parameters = json.load(f)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"
        self.logger.info("Device: %s" % self.device)
        raw_data = pd.read_csv(data_file_path)
        # 保留id方便后续追踪
        self.text_data, self.tag_data, self.extra_data, self.ids = self.prepare_data(raw_data)
        self.slice_number = [len(slices) for slices in self.text_data]
        self.rel_labels = self.init_relevance()
        self.data_len = len(self.text_data)
        self.logger.info("Data length: %d" % self.data_len)
        self.bert_path = "../Models/Bert/"
        # Do embedding
        self.judger_model = JudgeNet(self.bert_path).to(self.device)
        self.scorer_model = ScorerNet(self.bert_path, hidden_length=128).to(self.device)

    def init_relevance(self):
        max_text_length = self.parameters['max_text_length']
        rel_labels = []
        for text in self.text_data:
            total_len = 0
            text_len = len(text)
            # 正反标记，加起来等于长度则重复了
            idx = 0
            idx_rev = 0
            labels = [0] * text_len
            while True:
                if idx + idx_rev == text_len or total_len > max_text_length:
                    break
                labels[idx] = 1
                total_len += len(text[idx])
                idx += 1

                if idx + idx_rev == text_len or total_len > max_text_length:
                    break
                idx_rev += 1
                labels[-idx_rev] = 1
                total_len += len(text[-idx_rev])

            rel_labels.append(labels)
        return rel_labels


    def prepare_data(self, data:pd.DataFrame):
        text_data = data["separated_text"].apply(lambda text: json.loads(text))

        # 修剪分段，如果分段长度太长就只取头部，如果太短就舍弃，如果一条数据不存在合理的分段，那么舍弃这条数据
        # 同时进行Bert模型的tokenize
        tokenizer = BertTokenizer(vocab_file=self.bert_path + 'vocab.txt')  # 初始化分词器
        drop_slice_number = 0
        drop_text_number = 0
        trim_slice_number = 0
        dropped_rows = []
        for i, text in enumerate(text_data):
            parsed_text = []
            for text_slice in text:
                slice_length = len(text_slice)
                if slice_length < self.parameters['min_slice_length']:
                    drop_slice_number += 1
                    continue
                if slice_length > self.parameters['max_slice_length']:
                    trim_slice_number += 1
                    text_trimmed = text_slice[:self.parameters['max_slice_length']]
                    text_tokens = tokenizer.tokenize(text_trimmed)
                    parsed_text.append(tokenizer.convert_tokens_to_ids(text_tokens))
                else:
                    text_tokens = tokenizer.tokenize(text_slice)
                    parsed_text.append(tokenizer.convert_tokens_to_ids(text_tokens))
            if len(parsed_text) == 0:
                drop_text_number += 1
                dropped_rows.append(i)
            else:
                text_data[i] = parsed_text

        self.logger.info('%d slices dropped, %d slices trimmed, %d rows dropped' % (drop_slice_number, trim_slice_number, drop_text_number))
        data["separated_text"] = text_data
        data.drop(index=data.index[dropped_rows], inplace=True)
        self.logger.info('%d data remaining' % len(data))
        datalen = len(data)
        text_tokenized = data["separated_text"]
        imp = np.array(data["clk"])
        beh = np.array(data["bclk"])
        id = np.array(data["id"])

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

        self.logger.info(cut_points)
        advid_onehot = []
        one_hot_len = len(config.advids)
        have_advid_onehot = 0
        for i in range(datalen):
            advid = data['advid'][i]
            try:
                idx = config.advids.index(str(advid))
                one_hot = np.eye(one_hot_len, dtype=int)[idx]
                have_advid_onehot += 1
            except ValueError:
                one_hot = np.zeros(one_hot_len, dtype=int)
            advid_onehot.append(one_hot)
        self.logger.info('%d data have advid one-hot tag' % have_advid_onehot)
        return text_tokenized, tag_data, advid_onehot, ids

    def bert_embedding(self, bert_path, text_data):
        tokenizer = BertTokenizer(vocab_file=bert_path + "vocab.txt")  # 初始化分词器

        # 如果第一段文本有10小段，则记录为[0, 10]，该list元素数量比tag多1，tag[i]对应文本separate_points[i]:separates[i+1]
        separated_texts = []
        separated_points = [0]
        max_text_length = 512
        for slices in text_data:
            for text in slices:
                separated_texts.append(text)
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

        # 按照文本分割的embedding
        embeds_per_text = []
        for i in range(len(separated_points) - 1):
            embeds_per_text.append(embeds[separated_points[i]:separated_points[i+1]])
        return embeds_per_text

    def prepare_bert_input(self, text_data:list)->list:


    def run_model(self, mode="train", trainning_size=0.8, num_epoch=10, batch_size=200, lr=1e-2, **kwargs):

        def choose_slice(slices, max_len):
            indexes = np.arange(len(slices))
            np.random.shuffle(indexes)
            length = 0
            slices_select = []
            for i in indexes:
                slices_select.append(i)
                length += len(slices[i])
                if length > max_len:
                    break
            slices_select.sort()
            return slices_select

        def get_judger_input(indexes):
            cls = 101
            sep = 102
            # index
            random_slices_data = {
                'text_data': [],
                'labels': [],
                'seps': []
            }
            relevant_slices_data = {
                'text_data': [],
                'labels': [],
                'seps': []
            }
            for i in indexes:
                text_data = self.text_data[i]
                label_data = self.rel_labels[i]
                random_indexes = choose_slice(text_data, self.parameters['max_text_length'])
                text_data_random = [cls] + sum([text_data[j] + [sep] for j in random_indexes], [])
                random_slices_data['text_data'].append()
                lengths = [len(text_data[j]) for j in random_indexes]
                random_slices_data['labels'].append([label_data[j]])

                relevant_ones = []
                irrelevant_ones = []
                # 在irrelevant里随机选的index是irrelevant_ones的index，需要map
                irrelevant_matches = []
                total_len = 0
                for j, rele in enumerate(self.rel_labels[i]):
                    if rele == 1:
                        relevant_ones.append(j)
                        total_len += len(text_data[j])
                    else:
                        irrelevant_ones.append(text_data[j])
                        irrelevant_matches.append(j)
                irrelevant_indexes = choose_slice(irrelevant_ones, self.parameters['max_text_length'] - total_len)
                irrelevant_indexes = [irrelevant_matches[idx] for idx in irrelevant_indexes]
                relevant_ones += irrelevant_indexes
                relevant_ones.sort()
                relevant_slices.append(relevant_ones)
            cls = 101
            sep = 102
            random_slices = [slice + [sep] for slice in random_slices]
            relevant_slices = [slice + [sep] for slice in random_slices]
            zrand = [[cls] + sum(slices, []) for slices in random_slices]
            zrele = [[cls] + sum(slices, []) for slices in relevant_slices]
            return zrand, zrele, random_labels, relevant_labels

        def get_scorer_input(indexes, scores):



        self.logger.info("Running model, %s" % mode)

        judger_optim = torch.optim.Adam(self.judger_model.parameters(), lr=self.parameters['judger_lr'])
        scorer_optim = torch.optim.Adam(self.scorer_model.parameters(), lr=self.parameters['scorer_lr'])
        assert mode in ["train", "predict"]
        if mode == "train":
            indexes = np.arange(self.data_len)
            np.random.shuffle(indexes)
            train_len = round(self.data_len * trainning_size)
            train_inds = indexes[:train_len]
            self.logger.info('Training data length: %d' % len(train_inds))
            test_inds = indexes[train_len:]
            # 生成的训练、测试数据供测试使用
            # 取训练集的1/10
            train_inds_select = train_inds[::4]
            training_sample_input, training_sample_seps = build_model_input(text_embed_data, extra_feats, train_inds_select)
            training_sample_tags = tags[train_inds_select]

            testing_input, testing_seps = build_model_input(text_embed_data, extra_feats, test_inds)
            testing_tags = tags[test_inds]

            n_batch = math.ceil(len(train_inds) / batch_size)
            self.logger.debug("Batch number: %d" % n_batch)
            best_micro_f1 = 0
            best_epoch = 0
            for epoch in range(num_epoch):

                # if epoch % 10 == 0:
                #     self.logger.info('Epoch number: %d' % epoch)

                if epoch % 1 == 0:
                    cpc_pred_train, train_loss = self.model(training_sample_input, training_sample_tags.to(self.device), separates=training_sample_seps)
                    cpc_pred_test, test_loss = self.model(testing_input, testing_tags.to(self.device), separates=testing_seps, detail=True)
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
                    train_tags_cpu = training_sample_tags.cpu()
                    test_tags_cpu = testing_tags.cpu()
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
                    start = i * batch_size
                    # 别忘了这里用了sigmoid归一化
                    data_inds = train_inds[start : start + batch_size]
                    text_data = self.text_data[data_inds]
                    zrand, zrele, random_labels, relevant_labels = get_judger_input(data_inds)
                    rand_scores, rand_loss = self.judger_model()
                    data, seps = build_model_input(text_embed_data, extra_feats, data_inds)
                    _tags = tags[data_inds].to(self.device)
                    cpc_pred, loss = self.model(
                        data, _tags, seps
                    )  # text_hashCodes是一个32-dim文本特征
                    optimizer.zero_grad()
                    self.logger.debug(loss)
                    loss.backward()
                    for name, param in self.model.named_parameters():
                        self.logger.debug(param.grad)
                    optimizer.step()

                np.random.shuffle(train_inds)

                for name, param in self.model.named_parameters():
                    if name == "fcs.2.bias":
                        self.logger.debug(name, param)
        else:
            pass
            # n_batch = self.data_len // batch_size
            # for i in range(n_batch):
            #     start = i * batch_size
            #     # 别忘了这里用了sigmoid归一化
            #     cpc_pred = self.model(
            #         input_data[:, start : start + batch_size]
            #     )  # text_hashCodes是一个32-dim文本特征
            #
            #     loss = F.mse_loss(cpc_pred, tags)


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

def gen_correct_data(text_data, embeds):
    separated_points = [0]
    total_len = 0
    for slices in text_data:
        total_len += len(slices)
        separated_points.append(total_len)
    embeds_flat = sum(embeds, [])
    embeds_per_text = []
    for i in range(len(separated_points) - 1):
        embeds_per_text.append(embeds_flat[separated_points[i]:separated_points[i + 1]])
    return embeds_per_text

def main(log_level: str, data_path: str = "../Data/kuaishou_data_es.csv") -> None:
    # embed_data = np.load('../Data/vector_embed.npz')
    # dropped_data = embed_data['dropped_ind']
    d

    # 我觉得过一段时间就很难看懂了，不过还是很帅
    # process_mark = np.concatenate(
    #     [np.arange(1, 1 + sep_points[i + 1] - sep_points[i]) / (sep_points[i + 1] - sep_points[i])
    #      for i in range(len(sep_points) - 1)])
    # process_mark = process_mark.reshape([len(process_mark), -1])
    # extra_features = np.concatenate([advid_onehot, process_mark], axis=1)
    scorer.run_model(num_epoch=10000, trainning_size=0.8, extra_features=advid_onehot, batch_size=500, lr=1e-3, ids=id, text=text_data)


if __name__ == "__main__":
    # data_source raw embed
    # embed_type local api
    main(data_source="embed", embed_type="local", log_level=logging.INFO, data_path='../Data/kuaishou_data_0426.csv')
    # 以后就不需要转换embed了

