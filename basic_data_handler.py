#!/usr/bin/env python
# encoding: utf-8
"""
@author: zhangruifeng
@contact: zrf1999@pku.edu.cn
@file: basic_data_handler.py
@time: 2021/5/4 13:24
@github: local 16351726fa15c85f565b7d5fecdf320ea67a72ef
"""
from __future__ import print_function
import numpy as np
import pandas as pd
import logging
import os
import config
from matplotlib import pyplot as plt
from sklearn import linear_model
import json
import datetime
from utils import *
from multiprocessing import Process, Array, Value, Lock

import os

import scenedetect
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager, save_images
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

from scenedetect import VideoManager
from scenedetect import SceneManager

# For content-aware scene detection:
from scenedetect.detectors import ContentDetector

class DataHandler:
    def __init__(self, data_path, log_level=logging.INFO):
        self.data_path = data_path
        self.data_folder = os.path.split(data_path)[0]
        self.logger = init_logger(log_level, name="DataHandler")
        self.data = pd.read_csv(data_path)
        self.data_len = len(self.data)
        self.logger.info("Date length: %d" % self.data_len)
        self.data["key"] = list(
            map(
                lambda x1, x2: str(x1) + " " + str(x2),
                self.data["id"],
                self.data["advid"],
            )
        )

    def gen_tag(self, cols: list, thresholds, bin_num=config.bin_number):
        # 采用固定阈值的方法，生成标签
        if len(cols) == 1:
            col_name = cols[0]

            tag_data = []
            for i in self.data[col_name]:
                for j, thre in enumerate(thresholds):
                    if i > thre:
                        tag_data.append(j+1)
                        break
                    if j == len(thresholds) - 1:
                        tag_data.append(0)
            tag_data = np.array(tag_data)
            self.logger.info(
                "Tag positive count {}, negative count {}".format(
                    np.sum(tag_data == 1), np.sum(tag_data == 0)
                )
            )
            self.data["tag"] = tag_data
            aggs = self.data.groupby(["advid"], as_index=True)["tag"].agg(
                ["sum", "count"]
            )
            aggs = pd.DataFrame(aggs)
            # aggs.to_csv(config.grouped_data_file)
            aggs_dict = aggs.to_dict(orient="index")
            mean_vals = []
            for i, advid in enumerate(self.data["advid"]):
                grouped_data = aggs_dict[advid]
                mean_vals.append(grouped_data["sum"] / grouped_data["count"])

            self.data["mean_val"] = mean_vals
            aggs_tag = self.data.groupby(["advid"], as_index=True)[col_name].agg(
                ["median"]
            )
            aggs_tag = pd.DataFrame(aggs_tag)
            aggs_tag_dict = aggs_tag.to_dict(orient="index")
            mean_tag = []
            for tag_num, advid in zip(self.data[col_name], self.data["advid"]):
                grouped_data = aggs_tag_dict[advid]
                mean_tag.append(int(tag_num > grouped_data["median"]))
            self.data["mean_tag"] = mean_tag

        # 采取在广告主下做归一化以及直接使用比值，threshold定义有所不同
        else:
            threshold = thresholds[0]
            col_son, col_base = self.data[cols[0]], self.data[cols[1]]
            datalen = len(self.data)
            for i in range(datalen):
                if col_base[i] < threshold:
                    col_son[i] = 0
                if col_base[i] == 0:
                    col_base[i] = 1
            tag_data = np.array(col_son) / np.array(col_base)
            self.data["tag"] = tag_data
            binned_tag, cut_points = bin_tags(tag_data, bin_num)
            self.data["binned_tag"] = binned_tag
            self.logger.info(cut_points)
            aggs = self.data.groupby(["advid"], as_index=True)["tag"].agg(
                ["mean", "std", "count"]
            )
            aggs = pd.DataFrame(aggs)
            # aggs.to_csv(config.grouped_data_file)
            aggs_dict = aggs.to_dict(orient="index")
            # advid_mean = []
            # advid_std = []
            mean_tag = []
            removed_rows = []
            mean_vals = []
            for i, advid in enumerate(self.data["advid"]):
                grouped_data = aggs_dict[advid]
                if grouped_data["count"] < config.advid_threshold:
                    removed_rows.append(i)
                else:
                    bctr = self.data["tag"][i]
                    bctr_mean = grouped_data["mean"]
                    mean_vals.append(bctr_mean)
                    bctr_std = grouped_data["std"]
                    # advid_mean.append(bctr_mean)
                    # advid_std.append(bctr_std)
                    mean_tag.append((bctr - bctr_mean) / (bctr_std + 1e-10))
            self.logger.info(
                "%d data dropped due to advertiser cold start" % len(removed_rows)
            )
            binned_tag, cut_points = bin_tags(mean_tag, bin_num)
            self.logger.info(cut_points)
            self.data.drop(index=self.data.index[removed_rows], inplace=True)
            self.data["mean_tag"] = binned_tag
            self.data["mean_val"] = mean_vals
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
        separate_timestamps = []
        speech_speeds = []
        sum = 0
        for raw, id in zip(raw_data, ids):
            sentences = []
            timestamps = []
            raw = json.loads(raw)
            words_with_time = []
            speech_speed = []
            for res_detail in raw["Response"]["Data"]["ResultDetail"]:
                words_with_time.extend(res_detail["Words"])
                speech_speed.append(res_detail["SpeechSpeed"])
            speech_speeds.append(np.mean(speech_speed))
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
                if sentence == '':
                    timestamps.append([sta])
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
                        timestamps[-1].append(end)

                    # 不是必分，长度够了才分
                    if len(sentence) > 10:
                        sentences.append(sentence)
                        sentence = ""
                        timestamps[-1].append(end)

            for sep in sentences:
                if len(sep) > 50:
                    sum += 1
                    # print(sep, id)
                    # print(raw)
                seperated_len.append(len(sep))
            seps.append(len(sentences))
            separate_timestamps.append(json.dumps(timestamps))
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
        self.data["speech_speed"] = speech_speeds
        self.data["separate_timestamps"] = separate_timestamps
        # fig, subs = plt.subplots(2, 1)
        # subs[0].hist(seps, bins=10)
        #
        # subs[1].hist(seperated_len, bins=10)
        #
        # plt.show()

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
        print(
            "https://constrain.adwetec.com/material/creative/video/"
            + data["file"][row_num]
        )
        print(data.iloc[row_num])

    def build_sample(self, sample_number=1000):
        video_folder = config.video_folder
        ids = list(map(lambda x: x.split(".")[0], os.listdir(video_folder)))
        ids = list(filter(lambda x: not x.endswith("_origin"), ids))
        ids = [int(id) for id in ids]
        sep = len(ids) // sample_number
        selected_ids = ids[::sep]
        sample_data = self.data.loc[self.data["id"].isin(selected_ids)]
        sample_file = os.path.join(config.this_folder, "Data", "sample", "data.csv")
        sample_data.to_csv(sample_file)

    def display_tags(self):
        tags = np.array(self.data["tag"])
        bs = 1000
        nbt = len(tags) // bs
        tag_pos = tags == 1
        for i in range(nbt):
            print(np.sum(tag_pos[i * bs : (i + 1) * bs]) / bs)

    def check_column(self, colname):
        col_data = np.array(self.data[colname])
        ids = np.array(self.data["id"])
        print(ids[col_data == 1088])
        val_dict = {}
        for i in col_data:
            try:
                val_dict[i] += 1
            except:
                val_dict[i] = 1
        print(val_dict)
        print(col_data.mean(), col_data.max(), col_data.min(), col_data.std())

    def check_ids(self):
        id_groups = list(self.data["id"])
        print(len(id_groups), len(set(id_groups)))

    # 获取两个信息，第一个是视频上传次数，第二个是距离首次上传的时间
    def get_upload_times(self):
        group_ids = np.array(self.data["group_id"])
        ids = self.data["key"]
        upload_time = self.data["upload_time"]
        group_num = np.max(group_ids) + 1
        self.logger.info("Group number: %d" % group_num)
        relation_groups = []
        for i in range(group_num):
            relation_groups.append({})

        for i in range(self.data_len):
            id = ids[i]
            group_id = group_ids[i]
            time = datetime.datetime.strptime(upload_time[i], "%Y-%m-%d")
            relation_groups[group_id][id] = time

        zero_count = 0
        for i in range(group_num):
            sorted_upload = list(sorted(relation_groups[i].items(), key=lambda x: x[1]))
            if len(sorted_upload) > 0:
                first_upload = sorted_upload[0][1]
                for j in range(len(sorted_upload)):
                    time_delta = sorted_upload[j][1] - first_upload
                    sorted_upload[j] = (sorted_upload[j][0], (time_delta.days, j))
                relation_groups[i] = dict(sorted_upload)
            else:
                zero_count += 1
        self.logger.info("%d groups don't have data" % zero_count)

        times_already_uploaded = []
        days_to_first_upload = []
        for i in range(self.data_len):
            id = ids[i]
            group_id = group_ids[i]
            times_already_uploaded.append(relation_groups[group_id][id][1])
            days_to_first_upload.append(relation_groups[group_id][id][0])
        self.data["times_already_uploaded"] = times_already_uploaded
        self.data["days_to_first_upload"] = days_to_first_upload
        # group_number = group_number.value
        # for i in range(group_number+1):
        #     relation_groups.append({})
        #
        # for i in range(self.data_len):
        #     time = datetime.datetime.strptime(upload_time[i], "%Y-%m-%d")
        #     total_cost = cost[i]
        #     relation_groups[group_number][time] = total_cost
        #
        # for i in range(group_number + 1):
        #     if len(relation_groups[i]) > 1:
        #         sorted_upload = list(sorted(relation_groups[i].items(), key=lambda x: x[0]))
        #         first_upload = sorted_upload[0][0]
        #         for j in range(len(sorted_upload)):
        #             time_delta = sorted_upload[j][0] - first_upload
        #             sorted_upload[j] = (time_delta.days, sorted_upload[j][1])
        #         relation_groups[i] = sorted_upload
        # self.logger.info("Total groups: %d" % (group_number + 1))
        # relation_groups = np.array(relation_groups, dtype=object)
        # np.save(os.path.join(self.data_folder, config.grouped_data_file), relation_groups)


    def group_similar_texts(self, process_number):
        # self.data = self.data[0:100]
        # self.data_len = 100

        text = self.data["full_texts"]
        cost = self.data["cost"]
        group_ids = Array('i', np.zeros(self.data_len, dtype=int) - 1)
        upload_time = self.data["upload_time"]
        for i in upload_time:
            print(datetime.datetime.strptime(i, "%Y-%m-%d"))
        relation_groups = []
        group_number = Value('i', -1)

        process_inds = []
        for i in range(process_number):
            process_inds.append(list(range(i, self.data_len, process_number)))
        process_pool = []
        lock = Lock()
        for i in range(process_number):
            p = Process(target=worker, args=(text, self.data_len, process_inds[i], group_ids, group_number, i, lock))
            process_pool.append(p)
            p.start()
        for i in range(process_number):
            process_pool[i].join()
        self.data["group_id"] = group_ids
        self.store_data()

    def get_tag_in_advid(self):
        advids = self.data["advid"]
        all_match = 0
        for advid in set(list(advids)):
            matches = []
            for i in range(config.bin_number):
                match = sum(self.data["tag"][advids==advid]==i)
                matches.append(match)
                print("Advid %d, tag %d, sum %d" % (advid, i, match))
            all_match += np.max(matches)
        print(f"Precision {all_match / self.data_len}")

    def find_scenes(self, video_folder, threshold=50.0):
        video_paths = os.listdir(video_folder)
        # self.data.set_index("id", inplace=True)
        ids = self.data["id"]
        sep_times = {}
        for i, vp in enumerate(video_paths):
            if i % 100 == 0:
                self.logger.critical("Extracting scene frames, at %d/%d" % (i, len(video_paths)))
            id = vp.split('.')[0]
            # row = self.data.loc[int(id)]
            sep_times[id] = []
            video_path = os.path.join(video_folder, vp)

            # Create our videos & scene managers, then add the detector.
            video_manager = VideoManager([video_path], logger=self.logger)
            scene_manager = SceneManager()
            scene_manager.add_detector(
                ContentDetector(threshold=threshold))

            # Improve processing speed by downscaling before processing.
            # video_manager.set_downscale_factor(2)

            # Start the videos manager and perform the scene detection.
            video_manager.start()
            scene_manager.detect_scenes(frame_source=video_manager)

            scene_list = scene_manager.get_scene_list()
            # Each returned scene is a tuple of the (start, end) timecode.
            save_images(scene_list, video_manager, image_name_template='%s-Scene-$SCENE_NUMBER' % id,
                        num_images=1, output_dir=os.path.join(config.data_folder, 'scene_cover'))

            for scene in scene_list:
                sta_sec = scene[0].get_seconds()
                end_sec = scene[1].get_seconds()
                sep_times[id].append((sta_sec, end_sec))

            # print(id)
            # for i, scene in enumerate(scene_list):
            #     print('    Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
            #         i + 1,
            #         1000 * scene[0].get_seconds(), scene[0].get_frames(),
            #         1000 * scene[1].get_seconds(), scene[1].get_frames()))
            # separate_texts = json.loads(row["separated_text"])
            # separate_timestamps = json.loads(row["separate_timestamps"])
            # for text, timestamp in zip(separate_texts, separate_timestamps):
            #     print(text, timestamp)
            # print('-' * 30)
            video_manager.release()
        sep_times_list = [json.dumps(sep_times[str(id)]) for id in ids]
        self.data["scene_times"] = sep_times_list

        # return scene_list





def worker(data, data_len, indexes, group_ids, group_number, process_id, lock):
    for num, i in enumerate(indexes):
        if num % 100 == 0:
            print("Process: %d Doing similarity check, at %d/%d, found groups %d" % (process_id, num, len(indexes), group_number.value + 1))
        if group_ids[i] != -1:
            continue
        my_group = [i]
        t1 = data[i]
        for j in range(data_len):
            if group_ids[j] != -1:
                continue
            t2 = data[j]
            if similarity(t1, t2) > 0.9:
                my_group.append(j)
        if group_ids[i] != 1:
            lock.acquire()
            group_number.value += 1
            for j in my_group:
                group_ids[j] = group_number.value
            lock.release()

if __name__ == "__main__":
    # data_handler = DataHandler(config.raw_data_file)
    data_handler = DataHandler(config.raw_data_file, log_level=logging.FATAL)

    # data_handler.seperate_text()

    # data_handler.find_scenes(config.video_folder)
    #
    # data_handler.store_data()

    # data_handler.get_tag_in_advid()
    # data_handler.get_upload_times()
    # data_handler.store_data()

    # data_handler.group_similar_texts(8)
    # data_handler.store_data()

    # data_handler.check_column("width")

    # data_handler.check_data(126989)
    # data_handler.check_data(100764)

    # data_handler.display_tags()

    # data_handler.seperate_text()
    #
    data_handler.gen_tag(["cost"], [100, 1000])
    #
    data_handler.store_data()

    # data_handler.build_sample(1000)

    # data_handler.relations_bctr_imp(img=False)
