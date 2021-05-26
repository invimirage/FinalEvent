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
from sklearn import linear_model


class DataShower(DataHandler):
    def __init__(self, file_path, log_level):
        super().__init__(file_path, log_level)
        self.graph_prefix = "Data_"
        self.colors = ["cyan",
                       "lawngreen",
                       "orchid",
                       "deepskyblue",
                       "sandybrown",
                       "mintcream",
                       "coral"]

    def show_data_distribution(self, columns):
        if not isinstance(columns, list):
            columns = [columns]
        for column in columns:
            data = self.data[column]
            self.draw_histogram(data, column)

    # 计算advid分类得到的结果，与模型输出对比
    def compare_model_and_advid(self, id, pred):
        tag_data = self.data["tag"]
        aggs = self.data.groupby(["advid"], as_index=True)["tag"].agg(
            ["sum", "count"]
        )
        aggs = pd.DataFrame(aggs)
        aggs_dict = aggs.to_dict(orient="index")
        mean_vals = []
        for i, advid in enumerate(self.data["advid"]):
            grouped_data = aggs_dict[advid]
            mean_vals.append(grouped_data["sum"] / grouped_data["count"])
        for id in self.data["id"]:
            pass
        self.data["mean_val"] = mean_vals

    def cal_auc(self, pred, tag, triple_bins=True, keys=None):
        # 按照第-1维计算AUC
        if triple_bins:
            pred = np.array(pred)[:, 0] + 0.5 * np.array(pred)[:, 1]
            self.gen_tag(["cost"], "tag", [200], 2)
            self.set_keys()
            tags = []
            for key in keys:
                tag = self.data.loc[key, "tag"]
                tags.append(tag)
            tag = np.array(np.array(tags) == 0, dtype=int)
        else:
            pred = np.array(pred)[:, -1]
            tag = np.array(np.array(tag) == np.max(tag), dtype=int)

        fpr, tpr, thresholds = roc_curve(tag, pred, pos_label=1)
        plt.figure(figsize=(6, 6))
        plt.title('Validation ROC')
        plt.plot(fpr, tpr, 'b', label='Val AUC = %0.3f' % auc(fpr, tpr))
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

    def draw_histogram(self, data, title, xlabel, ylabel, binnum=15, color="deepskyblue"):
        data = np.array(data)
        fig_save_path = os.path.join(config.graph_folder, title + '.png')
        self.logger.info(f"{title}: mean {np.mean(data)}, max {np.max(data)}, min {np.min(data)}, std {np.std(data)}")
        # plt.title(f"title")
        plt.figure(figsize=(25, 15), dpi=100)
        plt.ylabel(ylabel, fontsize=18)  # 设置x轴，并设定字号大小
        plt.xlabel(xlabel, fontsize=18)  # 设置y轴，并设定字号大小
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        res = plt.hist(x=np.array(data), bins=binnum, color=color, rwidth=0.8)
        # for i in res:
        #     bin_height = i.get_height()
        #     plt.text(i.get_x()+i.get_width()/2, bin_height, "%d" % int(bin_height), fontsize=12, va="bottom", ha="center")
        plt.savefig(fig_save_path)
        plt.clf()


    def draw_bars(self, x, y, title, labels, total_width=0.8, horizontal=False):
        fig_save_path = os.path.join(config.graph_folder, title + '.png')
        if not isinstance(y[0], list):
            y = [y]
        bar_num = len(y)
        plt.figure(figsize=(20, 15), dpi=120)
        plt.title(title, fontsize=24, color="orangered")
        plt.ylabel('Tag', fontsize=18)  # 设置x轴，并设定字号大小
        plt.xlabel('Best micro-f1', fontsize=18)  # 设置y轴，并设定字号大小
        plt.yticks(range(len(x)), x, fontsize=14)
        plt.xlim(left=0.5, right=0.85)
        width = total_width / bar_num
        for this_bar in range(bar_num):
            if not horizontal:
                center = this_bar * width + 0.5 * width - total_width * 0.5
                x = plt.barh(np.arange(len(x)) + center, y[this_bar], height=width, color=self.colors[this_bar], label=labels[this_bar])
                for i in x:
                    height_x = i.get_width()
                    plt.text(height_x, i.get_y() + i.get_height() / 2, "%.3f" % height_x, fontsize=12, va="center", ha="left")
            else:
                pass
        # plt.barh(range(len(x)), y[0], height=1, color="orange",
        #              label=labels[0])
        plt.legend(loc="best", fontsize=18)
        plt.savefig(fig_save_path)
        plt.show()
        plt.clf()

    def regression(self, colx, coly):
        fig_save_path = os.path.join(config.graph_folder, f"Regression for {colx} and {coly}.png")
        datax = np.array(self.data[colx])
        datay = np.array(self.data[coly])
        rm = []
        for i in range(datay.shape[0]):
            if datax[i] > 200 or datay[i] > 200:
                rm.append(i)
        self.logger.info(len(rm))
        datay = np.delete(datay, rm)
        datax = np.delete(datax, rm)
        self.logger.info(len(datax))
        # 建立线性回归模型
        regr = linear_model.LinearRegression()
        # 拟合
        regr.fit(datax.reshape(-1, 1), datay)  # 注意此处.reshape(-1, 1)，因为X是一维的！
        # 不难得到直线的斜率、截距
        a, b = regr.coef_, regr.intercept_
        # 方式2：根据predict方法预测的价格
        # print(regr.predict(area))

        # 画图
        plt.figure(figsize=(20, 15), dpi=100)
        # plt.title(title, fontsize=24, color="orangered")
        plt.ylabel(colx, fontsize=18)  # 设置x轴，并设定字号大小
        plt.xlabel(coly, fontsize=18)  # 设置y轴，并设定字号大小
        # 1.真实的点
        plt.scatter(datax[::100], datay[::100], color='blue')

        # 2.拟合的直线
        plt.plot(datax, regr.predict(datax.reshape(-1, 1)), color='red', linewidth=4)
        plt.savefig(fig_save_path)
        plt.show()


# parses log file and model file
class ModelResults:
    def __init__(self, log_file_name, log_level):
        self.logger = init_logger(log_level, name="model_result_parser")
        self.log_file = os.path.join(config.log_dir, "log_" + log_file_name) + ".txt"
        self.model_file = os.path.join(config.model_save_path, log_file_name) + ".pth"

    def get_model_result(self):
        model_file = self.model_file
        f1, id, pred, tag = load_model(model_file, model=None, info=True)
        print(f1)
        return f1, id, pred, tag

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


def show_basic_status():
    graph_drawer = DataShower(config.raw_data_file, logging.INFO)
    duration = graph_drawer.data["duration"]
    seps = graph_drawer.data["separated_text"]
    sentence_len = []
    sep_num = []
    seped_len = []
    for i in seps:
        i = json.loads(i)
        seped_len.extend([len(x) for x in i])
        sep_num.append(len(i))
        sentence_len.append(sum([len(x) for x in i]))
    # graph_drawer.draw_histogram(duration, "Duration", "Duration/(s)", "Sum")
    graph_drawer.draw_histogram(sentence_len, "SentenceLength", "Sentence Length/(word)", "Sum", color=graph_drawer.colors[-1])
    graph_drawer.draw_histogram(sep_num, "SeparateNumber", "Separate Number", "Sum",
                                color=graph_drawer.colors[-2])
    graph_drawer.draw_histogram(seped_len, "SeparateLength", "Separate Length/(word)", "Sum",
                                color=graph_drawer.colors[-3])

# key不正确，暂时搁置
def check_3_bin_res():
    log_parser = ModelResults("Separated_LSTM_with_upload_time&Video_Embed_with_cost_no_mean_3_bins",
                              log_level=logging.INFO)
    graph_drawer = DataShower(config.raw_data_file, logging.INFO)
    _, ids, pred, tag = log_parser.get_model_result()
    print(ids[0])
    print(len(ids))
    graph_drawer.cal_auc(pred, tag, triple_bins=True, keys=ids)

def repeat_videos():
    graph_drawer = DataShower(config.raw_data_file, logging.INFO)
    grouped_data = graph_drawer.get_upload_times()
    graph_drawer.set_keys()
    grouped_data = list(sorted(grouped_data, key=lambda x: len(x), reverse=True))[0:10]
    for group in grouped_data:
        costs = []
        for key in group:
            cost = graph_drawer.data.loc[key, "cost"]
            costs.append(cost)
        print(costs)
    # print(grouped_data[0])

# 两个列之间的回归直线
def show_relations():
    col1 = "like"
    col2 = "negative"
    graph_drawer = DataShower(config.raw_data_file, logging.INFO)
    graph_drawer.regression(col1, col2)


def show_tag_relations():
    col1 = "mean_tag_like"
    col2 = "mean_tag_clk"
    col3 = "mean_tag_negative"
    graph_drawer = DataShower(config.raw_data_file, logging.INFO)
    d1, d2, d3 = graph_drawer.data[col1], graph_drawer.data[col2], graph_drawer.data[col3]
    # d3 = 1 - d3
    print(np.sum(((d1==d2) & (d2==d3)) / graph_drawer.data_len))


def draw_distri_graphs():
    cols = "bclk,pclk,cost,clk,imp,share,comment,like,follow,cancelfollow,report,block,negative,paly3s".split(',')
    graph_drawer = DataShower(config.raw_data_file, logging.INFO)
    graph_drawer.show_data_distribution(cols)

def tags_f1s():
    models = os.listdir(config.model_save_path)
    models_test = [model for model in models if model.startswith("mean_tag_") or model.startswith("tag_")]
    f1s = [load_model(os.path.join(config.model_save_path, model_name)) for model_name in models_test]
    pattern = "tag_(.*?)_test"
    tag_names = [re.findall(pattern, filename)[0] for filename in models_test]
    data_dict = {}
    for model_name, f1, tag_name in zip(models_test, f1s, tag_names):
        if tag_name not in data_dict:
            data_dict[tag_name] = [0, 0]
        if model_name.startswith("mean"):
            data_dict[tag_name][1] = f1
        if model_name.startswith("tag"):
            data_dict[tag_name][0] = f1
    data_list = list(sorted(data_dict.items(), key=lambda x: x[1][1], reverse=True))
    tags = [data[0] for data in data_list]
    f1s = [data[1][0] for data in data_list]
    f1s_mean = [data[1][1] for data in data_list]
    graph_drawer = DataShower(config.raw_data_file, logging.INFO)
    advid_f1s = []
    for tag in tags:
        advid_f1s.append(graph_drawer.get_advid_f1(tag))
    graph_drawer.draw_bars(tags, [f1s_mean, f1s, advid_f1s], title="F1 for different tags", labels=["normalized", "original", "advid"], total_width=0.9)

def show_model_result():
    logs = os.listdir(config.log_dir)
    logs = [log for log in logs if "mean_like" in log]
    graph_drawer = DataShower(config.raw_data_file, logging.INFO)
    for log in logs:
        pat = "log_(.*?).txt"
        log_name = re.findall(pat, log)[0]
        print(log_name)
        results = ModelResults(log_name, logging.INFO)
        F1, ids, pred, tag = results.get_model_result()
        graph_drawer.cal_auc(pred, tag, False, ids)
        print(results.get_best_test())



if __name__ == "__main__":
    # repeat_videos()
    # check_3_bin_res()
    # draw_distri_graphs()

    # tags_f1s()
    # show_tag_relations()
    # show_relations()
    # show_basic_status()
    show_model_result()
    # for model in os.listdir(config.model_save_path):
    #     if model.endswith('.pth'):
    #         log_parser = ModelResults(model.split('.')[0], log_level=logging.INFO)
    #         _, _, pred, tag = log_parser.get_model_result()
    #         log_parser.cal_auc(pred, tag)
    # print(log_parser.get_best_test())

