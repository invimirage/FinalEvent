#!/usr/bin/env python
# encoding: utf-8
"""
@author: zhangruifeng
@contact: zrf1999@pku.edu.cn
@file: basic_data_handler.py
@time: 2021/5/4 13:24
@github: local 16351726fa15c85f565b7d5fecdf320ea67a72ef
"""

import numpy as np
import pandas as pd
import logging
import os
import config
from matplotlib import pyplot as plt
from sklearn import linear_model
import json
from utils import *


class DataHandler:
    def __init__(self, data_path, log_level=logging.INFO):
        self.data_path = data_path
        self.logger = init_logger(log_level, name="DataHandler")
        self.data = pd.read_csv(data_path)
        self.data["key"] = list(
            map(
                lambda x1, x2: str(x1) + " " + str(x2),
                self.data["id"],
                self.data["advid"],
            )
        )

    def gen_tag(self, cols: list, threshold):
        # 采用固定阈值的方法，生成标签
        if len(cols) == 1:
            col_name = cols[0]

            tag_data = []
            for i in self.data[col_name]:
                if i < threshold:
                    tag_data.append(0)
                else:
                    tag_data.append(1)
            tag_data = np.array(tag_data)
            self.logger.info(
                "Tag positive count {}, negative count {}".format(
                    np.sum(tag_data == 1), np.sum(tag_data == 0)
                )
            )
            self.data["tag"] = tag_data

        # 采取在广告主下做归一化以及直接使用比值，threshold定义有所不同
        else:
            col_son, col_base = self.data[cols]
            datalen = len(self.data)
            for i in range(datalen):
                if col_base[i] < threshold:
                    col_son[i] = 0
                if col_base[i] == 0:
                    col_base[i] = 1
            tag_data = np.array(col_son) / np.array(col_base)
            self.data["tag"] = tag_data
            binned_tag, cut_points = bin_tags(tag_data, config.bin_number)
            self.data["binned_tag"] = binned_tag
            self.logger.info(cut_points)
            aggs = self.data.groupby(["advid"], as_index=True)["tag"].agg(
                ["mean", "std", "count"]
            )
            aggs = pd.DataFrame(aggs)
            aggs.to_csv(config.grouped_data_file)
            aggs_dict = aggs.to_dict(orient="index")
            # advid_mean = []
            # advid_std = []
            mean_tag = []
            removed_rows = []
            for i, advid in enumerate(self.data["advid"]):
                grouped_data = aggs_dict[advid]
                if grouped_data["count"] < config.advid_threshold:
                    removed_rows.append(i)
                else:
                    bctr = self.data["tag"][i]
                    bctr_mean = grouped_data["mean"]
                    bctr_std = grouped_data["std"]
                    # advid_mean.append(bctr_mean)
                    # advid_std.append(bctr_std)
                    mean_tag.append((bctr - bctr_mean) / (bctr_std + 1e-10))
            self.logger.info(
                "%d data dropped due to advertiser cold start" % len(removed_rows)
            )
            self.data.drop(index=self.data.index[removed_rows], inplace=True)
            self.data["mean_tag"] = mean_tag
            # self.data['bctr_mean'] = bctr_mean
            # self.data['bctr_std'] = bctr_std

    def seperate_text(self):
        data = self.data
        # text_data = data["full_texts"]
        raw_data = data["audio_text"]
        ids = data["id"]
        seperated_len = []
        seps = []
        seperated_text = []
        sum = 0
        for raw, id in zip(raw_data, ids):
            sentences = []
            raw = json.loads(raw)
            words_with_time = []
            for res_detail in raw["Response"]["Data"]["ResultDetail"]:
                words_with_time.extend(res_detail["Words"])
            word_num = len(words_with_time)
            sentence = ""
            for i in range(word_num):
                sta = words_with_time[i]["OffsetStartMs"]
                end = words_with_time[i]["OffsetEndMs"]
                word = words_with_time[i]["Word"]
                if i < word_num - 1:
                    next_sta = words_with_time[i + 1]["OffsetStartMs"]
                else:
                    next_sta = 0
                sentence += word
                # 考虑分句
                if int(next_sta) - int(end) > 20 or word in ["，", "。"]:
                    # 必分，查看长度
                    if i == word_num - 1 or word == "。":
                        # 长度太短，嫩加就加到上一句
                        if len(sentence) <= 10 and len(sentences) > 0:
                            sentences[-1] += sentence
                        else:
                            sentences.append(sentence)
                        sentence = ""

                    # 不是必分，长度够了才分
                    if len(sentence) > 10:
                        sentences.append(sentence)
                        sentence = ""

            for sep in sentences:
                if len(sep) > 50:
                    sum += 1
                    # print(sep, id)
                    # print(raw)
                seperated_len.append(len(sep))
            seps.append(len(sentences))
            seperated_text.append(json.dumps(sentences))
            # if len(sentences) == 0:
            #     print(id, text, raw)
            # if len(text) > 50:
            #     print(text)
        seperated_len = np.array(seperated_len)
        seps = np.array(seps)
        self.logger.info("Super long text slice number: %d" % sum)
        self.logger.info(
            "Slice length, min %d, mean %.2lf, max %d"
            % (seperated_len.min(), seperated_len.mean(), seperated_len.max())
        )
        self.logger.info(
            "Slice number, min %d, mean %.2lf, max %d"
            % (seps.min(), seps.mean(), seps.max())
        )
        self.data["separated_text"] = seperated_text
        # fig, subs = plt.subplots(2, 1)
        # subs[0].hist(seps, bins=10)
        #
        # subs[1].hist(seperated_len, bins=10)
        #
        # plt.show()

    #
    # def cal_bctr_avg(self, img=False):
    #     data = self.data
    #     bctr_avg = data.groupby(['advid'], as_index=True)['like_rate'].agg(['mean', 'std', 'count'])
    #     bctr_avg = pd.DataFrame(bctr_avg)
    #     bctr_avg.to_csv(config.grouped_data_file)
    #     advid_bctr_dict = bctr_avg.to_dict(orient='index')
    #     if img:
    #         plt.hist(bctr_avg['mean'], bins=10)
    #         plt.show()
    #     advid_mean = []
    #     advid_std = []
    #     bctr_tag = []
    #     removed_rows = []
    #     for i, advid in enumerate(data['advid']):
    #         grouped_data = advid_bctr_dict[advid]
    #         if grouped_data['count'] < config.advid_threshold:
    #             removed_rows.append(i)
    #         else:
    #             bctr = data['like_rate'][i]
    #             bctr_mean = grouped_data['mean']
    #             bctr_std = grouped_data['std']
    #             advid_mean.append(bctr_mean)
    #             advid_std.append(bctr_std)
    #             bctr_tag.append((bctr - bctr_mean) / (bctr_std + 1e-10))
    #     self.logger.info('%d data dropped due to advertiser cold start' % len(removed_rows))
    #     self.data.drop(index=self.data.index[removed_rows], inplace=True)
    #     self.data['like_tag'] = bctr_tag
    #     self.data['bctr_mean'] = bctr_mean
    #     self.data['bctr_std'] = bctr_std

    def store_data(self):
        self.data.to_csv(self.data_path)

    def relations_bctr_imp(self, img):
        bctr = np.array(self.data["tag"])[::100]
        imp = np.log(np.array(self.data["clk"])[::100] + 1e-5)
        regr = linear_model.LinearRegression()
        regr.fit(imp.reshape(-1, 1), bctr)
        if img:
            plt.scatter(imp, bctr)
            plt.plot(imp, regr.predict(imp.reshape(-1, 1)), color="red", linewidth=4)
            plt.show()

    def check_data(self, data_id):
        data = self.data
        row_num = list(data["id"]).index(int(data_id))
        print("https://constrain.adwetec.com/material/creative/video/" + data['file'][row_num])

    def build_sample(self, sample_number=1000):
        sample_data = self.data.head(sample_number)
        sample_file = os.path.join(config.this_folder, 'Data', 'sample', 'data.csv')
        sample_data.to_csv(sample_file)

    def display_tags(self):
        tags = np.array(self.data['tag'])
        bs = 1000
        nbt = len(tags) // bs
        tag_pos = (tags==1)
        for i in range(nbt):
            print(np.sum(tag_pos[i*bs:(i+1)*bs]) / bs)


if __name__ == "__main__":
    # data_handler = DataHandler(os.path.join(config.data_folder, config.raw_data_file))

    data_handler = DataHandler(config.raw_data_file)
    data_handler.check_data(541132)
    # data_handler.display_tags()
    # data_handler.seperate_text()

    # data_handler.gen_tag(["cost"], 100)
    # data_handler.store_data()
    # data_handler.build_sample(1000)
    # data_handler.relations_bctr_imp(img=False)
