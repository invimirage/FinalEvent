#!/usr/bin/env python
# encoding: utf-8
"""
@author: zhangruifeng
@contact: zrf1999@pku.edu.cn
@file: draw_graph.py
@time: 2021/5/10 15:37
@github: local 16351726fa15c85f565b7d5fecdf320ea67a72ef
"""
from utils import *
import logging
import config
import pandas as pd
import os
from matplotlib import pyplot as plt
import json
import re
from basic_data_handler import DataHandler
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


class DataShower(DataHandler):
    def __init__(self, file_path, log_level):
        super().__init__(file_path, log_level)
        self.data = pd.read_csv(file_path)
        self.graph_prefix = "Data_"


class ModelResults:
    def __init__(self, log_file_name, log_level):
        self.logger = init_logger(log_level, name="model_result_parser")
        self.log_file = os.path.join(config.log_dir, "log_" + log_file_name) + ".txt"
        self.model_file = os.path.join(config.model_save_path, log_file_name) + ".pth"

    def get_model_result(self):
        model_file = self.model_file
        f1, id, pred, tag = load_model(model_file, model=None, info=True)
        return f1, id, pred, tag

    def cal_auc(self, pred, tag):
        # 按照第-1维计算AUC
        pred = np.array(pred)
        tag = np.array(np.array(tag) == np.max(tag), dtype=int)

        fpr, tpr, thresholds = roc_curve(tag, pred, pos_label=1)
        print("-----sklearn:", auc(fpr, tpr))

    # 加载文件，并选出f1最好的一次测试
    def get_best_test(self):
        with open(self.log_file, "r") as f:
            lines = f.readlines()

        best_lines = []
        best_f1 = 0
        sample_lines = []
        best_f1_a_sample = 0
        for line in lines:
            line = line.strip("\n")
            if "Best Micro-F1" in line:
                sta = line.index("Best Micro-F1: ") + len("Best Micro-F1: ")
                end = line.index(", epoch")
                f1 = float(line[sta:end])
                if f1 > best_f1_a_sample:
                    best_f1_a_sample = f1
            if "Running sample" in line and len(sample_lines) > 0:
                if best_f1_a_sample > best_f1:
                    best_f1 = best_f1_a_sample
                    best_lines = sample_lines
                sample_lines = []
                best_f1_a_sample = 0
            sample_lines.append(line)

        epoch_data = {}
        train_f1s = []
        # 标记位置
        train_f1_ind = -1
        train_losses = []
        train_loss_ind = -1
        test_f1s = []
        test_f1_ind = -1
        test_losses = []
        test_loss_ind = -1
        for i, line in enumerate(best_lines):
            line = line.strip("\n")
            if "Running sample" in line:
                after_which = "with parameters: "
                start_index = line.index(after_which) + len(after_which)
                json_str = line[start_index:].replace("'", '"')
                params = json.loads(json_str)
                epoch_data["params"] = params
            if "Training set" in line:
                train_f1_ind = i + 4
                train_loss_ind = i + 1
            if "Testing set" in line:
                test_f1_ind = i + 4
                test_loss_ind = i + 1
            if i == train_loss_ind:
                start_index = line.index("Loss: ") + len("Loss: ")
                train_losses.append(float(line[start_index:]))
            if i == test_loss_ind:
                start_index = line.index("Loss: ") + len("Loss: ")
                test_losses.append(float(line[start_index:]))
            if i == train_f1_ind:
                pattern = r"[[](.*?)[]]"
                f1s = list(
                    filter(lambda x: x != "", re.findall(pattern, line)[0].split(" "))
                )
                f1s = np.array(f1s, dtype=float)
                train_f1s.append(f1s.mean())
            if i == test_f1_ind:
                pattern = r"[[](.*?)[]]"
                f1s = list(
                    filter(lambda x: x != "", re.findall(pattern, line)[0].split(" "))
                )
                f1s = np.array(f1s, dtype=float)
                test_f1s.append(f1s.mean())
        epoch_data["train_f1"] = train_f1s
        epoch_data["test_f1"] = test_f1s
        epoch_data["train_loss"] = train_losses
        epoch_data["test_loss"] = test_losses
        return epoch_data


if __name__ == "__main__":
    for model in os.listdir(config.model_save_path):
        if model.endswith('.pth'):
            log_parser = ModelResults(model.split('.')[0], log_level=logging.INFO)
            _, _, pred, tag = log_parser.get_model_result()
            log_parser.cal_auc(pred, tag)
    # print(log_parser.get_best_test())

