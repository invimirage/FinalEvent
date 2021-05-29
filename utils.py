#!/usr/bin/env python
# encoding: utf-8
"""
@author: zhangruifeng
@contact: zrf1999@pku.edu.cn
@file: utils.py
@time: 2021/4/12 15:46
@github: local 16351726fa15c85f565b7d5fecdf320ea67a72ef
"""
import pandas as pd
import numpy as np
import logging
import config
import os
import torch
from models import *
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split


def bin_tags(tags, binnum):
    # 将数据映射到所需数量的分位数
    tags_binned = pd.qcut(tags, binnum, labels=False)
    # 按照指定的数值分桶
    # tags_binned = pd.cut(tags, [0, 0.0032, 1], labels=False)
    # 计算指定分位数点的数据
    large_counts_series = pd.Series(tags)
    cut_points = large_counts_series.quantile(np.linspace(0, 1, binnum + 1))
    return tags_binned, cut_points


def init_logger(log_level, name, write_to_file=False, clean_up=False):
    # logging.basicConfig(
    #     format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S"
    # )
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # 再创建一个handler，用于输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(log_level)

    ch.setFormatter(formatter)

    logger.addHandler(ch)
    if write_to_file:
        # 2、创建一个handler，用于写入日志文件
        log_dir = config.log_dir
        log_file = "log_%s.txt" % name
        log_path = os.path.join(log_dir, log_file)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if clean_up:
            fh = logging.FileHandler(log_path, mode="w")
        else:
            fh = logging.FileHandler(log_path, mode="a")
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


from collections import Iterable
from copy import deepcopy

number = 0


def sample_hyperparams(model_params, sample_list, test_params: dict) -> list:
    global number
    if len(test_params) == 0:
        if sample_list[number]:
            number += 1
            return [deepcopy(model_params)]
        else:
            number += 1
            return []
    sample_parameters = []
    for param_name in test_params:
        for param in test_params[param_name]:
            if param_name in model_params.keys():
                model_params[param_name] = param
            else:
                model_params["common"][param_name] = param
            other_params = deepcopy(test_params)
            del other_params[param_name]
            results = sample_hyperparams(model_params, sample_list, other_params)
            if len(results) > 0:
                sample_parameters.extend(results)
        break
    return sample_parameters


def adjust_hyperparams(logger, params, sample_number, model_name, run_model, **kwargs):
    global number
    if model_name == "SeparatedLSTM" or model_name == "BiLSTMWithAttention":
        # 调试slstm
        lr_high = 2
        lr_seps = 10
        lr_low = 5
        step = (lr_high - lr_low) / (lr_seps - 1)
        lrs = 10 ** (-(np.arange(lr_low, lr_high, step)))

        all_params = kwargs.get(
            "test_params",
            {
                "hidden_length": [128, 256, 512],
                "layer_number": [1, 2, 3],
                "linear_hidden_length": [64, 128],
                "drop_out_rate": [0.5, 0.6],
                "batch_size": [400, 600, 800],
                "learning_rate": 1e-4,
                "training_size": 0.8,
                "number_epochs": 100,
            },
        )
    elif model_name == "DoubleNet":
        all_params = kwargs.get(
            "test_params",
            {
                "hidden_length": [128, 256, 512],
                # "grad_layer_name": "encoder.layer.23.attention.self.query.weight",
                "drop_out_rate": 0.5,
                "batch_size": [400, 800, 1200],
                "learning_rate": 1e-4,
                "training_size": 0.8,
                "number_epochs": 100,
            },
        )
    elif model_name == "BertWithCNN":
        # 调试cnn
        lr_high = 2
        lr_seps = 10
        lr_low = 5
        step = (lr_high - lr_low) / (lr_seps - 1)
        lrs = 10 ** (-(np.arange(lr_low, lr_high, step)))
        all_params = kwargs.get(
            "test_params",
            {
                "hidden_length": [32, 16, 8],
                "linear_hidden_length": [32, 64, 128],
                # "grad_layer_name": "encoder.layer.23.attention.self.query.weight",
                "drop_out_rate": 0.5,
                "channels": [16, 32, 64],
                "batch_size": [50, 100, 150],
                "learning_rate": 1e-4,
                "training_size": 0.8,
                "number_epochs": 100,
            },
        )
    elif model_name == "BertWithMLP":
        # 调试cnn
        lr_high = 2
        lr_seps = 10
        lr_low = 5
        step = (lr_high - lr_low) / (lr_seps - 1)
        lrs = 10 ** (-(np.arange(lr_low, lr_high, step)))
        all_params = kwargs.get(
            "test_params",
            {
                "linear_hidden_length": [32, 64, 128],
                "grad_layer_name": "encoder.layer.23.attention.self.query.weight",
                "drop_out_rate": 0.5,
                "batch_size": [50, 100, 150],
                "learning_rate": 1e-4,
                "training_size": 0.8,
                "number_epochs": 100,
            },
        )
    elif model_name == "VideoNet":
        lr_high = 2
        lr_seps = 10
        lr_low = 5
        step = (lr_high - lr_low) / (lr_seps - 1)
        lrs = 10 ** (-(np.arange(lr_low, lr_high, step)))
        all_params = kwargs.get(
            "test_params",
            {
                "frame_rate": 1,
                "input_length": 512,
                "pic_linear_hidden_length": 1024,
                "pic_grad_layer_name": "aaa_blocks.31",
                "pic_drop_out_rate": [0.5, 0.3, 0.6],
                "pic_channels": 3,
                "layer_number": 3,
                "hidden_length": 512,
                "linear_hidden_length": 128,
                "drop_out_rate": 0.5,
                "batch_size": 400,
                "learning_rate": lrs,
                "training_size": 0.8,
                "number_epochs": 100,
                "random": 1,
            },
        )
    elif model_name == "VideoNetEmbed":
        lr_high = 2
        lr_seps = 10
        lr_low = 5
        step = (lr_high - lr_low) / (lr_seps - 1)
        lrs = 10 ** (-(np.arange(lr_low, lr_high, step)))
        all_params = kwargs.get(
            "test_params",
            {
                "frame_rate": 1,
                "input_length": 512,
                "pic_linear_hidden_length": 1024,
                "pic_grad_layer_name": "aaa_blocks.31",
                "pic_drop_out_rate": [0.5, 0.3, 0.6],
                "pic_channels": 3,
                "layer_number": 3,
                "hidden_length": 512,
                "linear_hidden_length": 128,
                "drop_out_rate": 0.5,
                "batch_size": 400,
                "learning_rate": lrs,
                "training_size": 0.8,
                "number_epochs": 100,
                "random": 1,
            },
        )
    elif model_name == "JointNet":
        all_params = kwargs.get(
            "test_params",
            {
                "text_hidden_length": [128, 256, 512],
                "text_layer_number": [1, 2, 3],
                "text_linear_hidden_length": [64, 128],
                "text_drop_out_rate": [0.5, 0.6],
                "text_batch_size": [400, 600, 800],
                "text_learning_rate": 1e-4,
                "text_training_size": 0.8,
                "text_number_epochs": 100,
                "video_frame_rate": 1,
                "video_input_length": 512,
                "video_pic_linear_hidden_length": 1024,
                "video_pic_grad_layer_name": "aaa_blocks.31",
                "video_pic_drop_out_rate": [0.5, 0.3, 0.6],
                "video_pic_channels": 3,
                "video_layer_number": 3,
                "video_hidden_length": 512,
                "video_linear_hidden_length": 128,
                "video_drop_out_rate": 0.5,
                "batch_size": 400,
                "learning_rate": 1e-5,
                "training_size": 0.8,
                "number_epochs": 100,
                "random": 1,
            },
        )
    elif model_name == "VideoAttention":
        all_params = kwargs.get(
            "test_params",
            {
                "linear_length": 512,
                "linear_hidden_length": 128,
                "frames_per_clip": 16,
                "grad_layer_name": "aaa_blocks.31",
                "img_size": 180,
                "drop_out_rate": 0.5,
                "batch_size": 400,
                "learning_rate": 1e-5,
                "training_size": 0.8,
                "number_epochs": 100,
                "random": 1,
            }
        )
    params_to_test = {}
    for param in all_params:
        if isinstance(all_params[param], Iterable) and not isinstance(
            all_params[param], str
        ):
            params_to_test[param] = all_params[param]
    samples = sample_number
    total_test_cases = 1
    for param_list in params_to_test.values():
        total_test_cases *= len(param_list)
    sample_rate = samples / total_test_cases
    is_sample = [True] * samples + [False] * (total_test_cases - samples)
    np.random.shuffle(is_sample)
    print(
        "In total {} test cases, sample rate {:.2%}, sample number {}".format(
            total_test_cases, sample_rate, samples
        )
    )
    number = 0
    sample_params = sample_hyperparams(
        params[model_name],
        is_sample,
        params_to_test,
    )
    for i, model_param in enumerate(sample_params):
        logger.info("Running sample {}, with parameters: {}".format(i, model_param))
        if model_name == "SeparatedLSTM" or model_name == "BiLSTMWithAttention":
            model = SeparatedLSTM(
                input_length=1024,
                extra_length=config.extra_feat_length,
                hyperparams=model_param,
            )
        elif model_name == "BiLSTMWithAttention":
            model = BiLSTMWithAttention(
                input_length=1024,
                extra_length=config.extra_feat_length,
                hyperparams=model_param,
            )
        elif model_name == "BertWithMLP":
            model = BertWithMLP(
                bert_path=config.bert_path,
                extra_length=config.extra_feat_length,
                hyperparams=model_param,
            )
        elif model_name == "BertWithCNN":
            model = BertWithCNN(
                bert_path=config.bert_path,
                extra_length=config.extra_feat_length,
                hyperparams=model_param,
            )
        elif model_name == "DoubleNet":
            model = DoubleNet(
                # 多一维位置信息
                input_length=1025 + config.extra_feat_length,
                hyperparams=model_param
            )
        elif model_name == "VideoNet":
            model = VideoNet(
                extra_length=config.extra_feat_length, hyperparams=model_param
            )
        elif model_name == "VideoNetEmbed":
            model = VideoNetEmbed(
                extra_length=config.extra_feat_length, hyperparams=model_param
            )
        elif model_name == "VideoAttention":
            model = VideoAttention(extra_length=config.extra_feat_length, hyperparams=model_param)
        elif model_name == "JointNet":
            hparams = {
                "video": {},
                "text": {},
                "common": model_param["common"]
            }
            for param, val in model_param.items():
                if param.startswith('video'):
                    striplen = len('video_')
                    hparams["video"][param[striplen:]] = val
                elif param.startswith('text'):
                    striplen = len('text_')
                    hparams["text"][param[striplen:]] = val
                else:
                    continue
            model = JointNet(
                input_length=1024, extra_length=config.extra_feat_length, hyperparams=hparams
            )
        run_model(model=model, logger=logger, kwargs=kwargs)

def output_logs(self, epoch, kwargs: dict, *args):
    train_pred, train_loss, train_inds, test_pred, test_loss, test_inds = args
    pred_worst = test_pred.cpu().detach().numpy()[:, 0].flatten()
    top10 = test_inds[np.array(pred_worst).argsort()[::-1][0:10]]
    for id, tag, text in zip(
        kwargs["ids"][top10], self.tag[top10], kwargs["text"][top10]
    ):
        self.logger.info("{} {} {}".format(id, tag, text))
    pred_best = test_pred.cpu().detach().numpy()[:, -1].flatten()
    top10 = test_inds[np.array(pred_best).argsort()[::-1][0:10]]
    self.logger.info("Best Top 10: {}".format(kwargs["ids"][top10]))
    for id, tag, text in zip(
        kwargs["ids"][top10], self.tag[top10], kwargs["text"][top10]
    ):
        self.logger.info("{} {} {}".format(id, tag, text))
    pred_train = np.argmax(train_pred.cpu().detach(), axis=1)
    pred_test = np.argmax(test_pred.cpu().detach(), axis=1)
    train_tags_cpu = self.tag[train_inds].cpu().numpy()
    test_tags_cpu = self.tag[test_inds].cpu().numpy()
    self.logger.info("------------Epoch %d------------" % epoch)
    self.logger.info("Training set")
    self.logger.info("Loss: %.4lf" % train_loss.cpu().detach())
    p_class, r_class, f_class, _ = precision_recall_fscore_support(
        pred_train, train_tags_cpu
    )
    self.logger.info(p_class)
    self.logger.info(r_class)
    self.logger.info(f_class)
    self.logger.info("Testing set")
    self.logger.info("Loss: %.4lf" % test_loss.cpu().detach())
    p_class, r_class, f_class, _ = precision_recall_fscore_support(
        pred_test, test_tags_cpu
    )
    self.logger.info(p_class)
    self.logger.info(r_class)
    self.logger.info(f_class)
    f1_mean = np.mean(f_class)
    return f1_mean


# 用于划分测试集和训练集
def sep_train_test(data_length, tag_data, training_size):
    data_indexes = np.arange(data_length)
    tag_data_numpy = tag_data.numpy()
    # 保持所有测试的训练集测试集相同
    train_inds, test_inds, _, _ = train_test_split(
        data_indexes, tag_data_numpy, test_size=1 - training_size, random_state=10086
    )
    return train_inds, test_inds


def build_batch(
    data_indexes: np.ndarray,
    batch_size: int,
    random=True,
    tag_data: torch.tensor = None,
):
    bin_number = config.bin_number
    n_batch = math.ceil(len(data_indexes) / batch_size)
    batch_data = []
    if random:
        if tag_data.shape[0] != data_indexes.shape[0]:
            tag_data = tag_data[data_indexes].numpy()
        else:
            tag_data = tag_data.numpy()
        each_batch_size = batch_size // bin_number + 1
        indexes_in_bins = []
        for i in range(bin_number):
            indexes_in_bins.append(data_indexes[tag_data == i])
            # print(len(indexes_in_bins[i]))
        for i in range(n_batch):
            this_batch_data = [
                np.random.choice(indexes_in_bins[j], each_batch_size, replace=False)
                for j in range(bin_number)
            ]
            this_batch_data = np.concatenate(this_batch_data, axis=0)
            batch_data.append(this_batch_data)
    else:
        np.random.shuffle(data_indexes)
        for i in range(n_batch):
            sta = i * batch_size
            end = (i + 1) * batch_size
            batch_data.append(data_indexes[sta:end])
    return batch_data

def save_the_best(model, f1, id, tag, pred, file_name):
    print(f"Saving model, F1 {f1}")
    save_path = os.path.join(config.model_save_path, file_name + ".pth")
    save_dict = {
        "model": model.state_dict(),
        "f1": f1,
        "id": np.array(id),
        "pred": pred.cpu().detach().numpy(),
        "tag": tag.cpu().detach().numpy(),
    }
    torch.save(save_dict, save_path, _use_new_zipfile_serialization=False)


def load_model(file_name, model=None, info=False):
    if not file_name.endswith("pth"):
        model_path = os.path.join(config.model_save_path, file_name + ".pth")
    else:
        model_path = file_name
    # print(model_path)
    try:
        checkpoint = torch.load(model_path)
    except Exception as err:
        print(err)
        return 0
    if model is not None:
        model.load_state_dict(checkpoint["model"])
    f1 = checkpoint["f1"]
    if not info:
        return f1
    else:
        id = checkpoint["id"]
        pred = checkpoint["pred"]
        tag = checkpoint["tag"]
        return f1, id, pred, tag


def normalization(data: np.ndarray, method="mean-std-sigmoid"):
    assert method in ["min-max", "mean-std-sigmoid"]
    if method == "min-max":
        data = (data - data.min()) / (data.max() - data.min())
    else:
        data = (data - data.mean()) / data.std()
        data = 1 / (1 + np.exp(-data))
    return data


# 生成除视频、文本特征以外的特征
def parse_extra_features(data: pd.DataFrame):
    duration = np.array(data["duration"])
    text_len = np.array([len(text) for text in list(data["full_texts"])])
    height = np.array(data["height"])
    width = np.array(data["width"])
    times_already_uploaded = np.array(data["times_already_uploaded"])
    days_to_first_upload = np.array(data["days_to_first_upload"])
    speech_speed = np.array(data["speech_speed"])
    # print(speech_speed.mean(), speech_speed.max(), speech_speed.min())
    is_vertical = np.array(height > width, dtype=int)
    # print(np.sum(is_vertical==1), np.sum(is_vertical==0))
    is_high_defination = (height * is_vertical + width * (1 - is_vertical)) >= 1920
    is_high_defination = np.array(is_high_defination, dtype=int)
    # print(np.sum(is_high_defination == 1), np.sum(is_high_defination == 0))
    duration = normalization(duration)
    text_len = normalization(text_len)
    speech_speed = normalization(speech_speed)
    times_already_uploaded = normalization(times_already_uploaded)
    days_to_first_upload = normalization(days_to_first_upload)
    advid_onehot = []
    one_hot_len = len(config.advids)
    advid_dict = {k: v for v, k in enumerate(config.advids)}
    for advid in data["advid"]:
        try:
            idx = advid_dict[str(advid)]
            one_hot = np.eye(one_hot_len, dtype=int)[idx]
        except KeyError:
            one_hot = np.zeros(one_hot_len, dtype=int)
        advid_onehot.append(one_hot)
    advid_onehot = np.array(advid_onehot)
    extra_features = np.column_stack(
        (
            is_high_defination,
            is_vertical,
            duration,
            text_len,
            speech_speed,
            advid_onehot,
            times_already_uploaded,
            days_to_first_upload
        )
    )
    return extra_features

from difflib import SequenceMatcher#导入库
def similarity(a, b):
    return SequenceMatcher(lambda x: x in [" ", "，", "。"], a, b).quick_ratio()#引用ratio方法，返回序列相似性的度量

import tracemalloc
def get_memory_status(line_number):
    pass
    # tracemalloc.start()
    # cur, _ = tracemalloc.get_traced_memory()
    # print(f"Line {line_number}: Current memory usage is {cur / 10**9}GB")
    # tracemalloc.stop()