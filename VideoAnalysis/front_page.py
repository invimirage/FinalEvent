#!/usr/bin/env python
# encoding: utf-8
"""
@author: zhangruifeng
@contact: zrf1999@pku.edu.cn
@file: front_page.py
@time: 2021/4/23 14:28
@github: local 16351726fa15c85f565b7d5fecdf320ea67a72ef
"""

import os
import cv2
import pandas as pd
import numpy as np
import torch
import logging
import math
import gc
from FinalEvent import config
from FinalEvent import models
from sklearn.metrics import precision_recall_fscore_support
from FinalEvent import utils

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")


class cover_parser:
    def __init__(self, image_folder_or_file, data_file, **kwargs):
        self.logger = logging.getLogger("MyLogger")
        self.logger.setLevel(kwargs["log_level"])
        image_dict = self.load_image(image_folder_or_file)
        self.image_data, self.tag_data, self.ids = self.get_tags_features(
            data_file, image_dict
        )
        self.model = models.PictureNet()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info("Device: %s" % self.device)

        # print(pictures[0:10])

    def load_image(self, image_folder_or_file):
        try:
            image_dict = np.load(image_folder_or_file, allow_pickle=True)
            image_dict = image_dict.item()
        except:
            image_list = list(
                filter(lambda x: x[-4:] == "jpeg", os.listdir(image_folder_or_file))
            )
            image_dict = {}
            self.logger.info("Loading image...")
            total_len = len(image_list)
            for i, img in enumerate(image_list):
                if i % 100 == 0:
                    self.logger.info("%d/%d" % (i, total_len))
                id = img.split(".")[0]
                image_path = os.path.join(image_folder_or_file, img)
                image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
                image = cv2.resize(image, (180, 320), interpolation=cv2.INTER_LANCZOS4)
                image = np.transpose(image, (2, 0, 1))
                # Deal with png
                image_dict[id] = image[0:3]
            np.save("../Data/image_data.npy", image_dict)
        return image_dict

    def get_tags_features(self, data_file, image_dict):
        data = pd.read_csv(data_file)
        imp = np.array(data["clk"])
        beh = np.array(data["bclk"])
        ids = np.array(data["id"])

        datalen = len(data)
        for i in range(datalen):
            if imp[i] < config.threshold:
                beh[i] = 0
            # 避免分母为0
            if imp[i] == 0:
                imp[i] = 1
        tags = beh / imp
        binned_tags, cut_points = utils.bin_tags(tags, config.bin_number)
        self.logger.info("Cut points :{}".format(cut_points))
        image_data = []
        tag_data = []
        ids_data = []
        for id, tag in zip(ids, binned_tags):
            try:
                # Deal with png
                image_data.append(image_dict[str(id)][0:3])
                # if image_data[-1].shape != (3, 320, 180):
                #     print(image_data[-1].shape)
                tag_data.append(tag)
                ids_data.append(id)
            except KeyError:
                continue
        self.logger.info(
            "Image data length: %d, image shape %s"
            % (len(image_dict), str(image_data[-1].shape))
        )
        self.logger.info("Tag data length: %d" % datalen)
        self.logger.info("Matched data length: %d" % len(tag_data))
        del image_dict
        gc.collect()
        tag_data = np.array(tag_data)
        image_data = np.array(image_data)
        ids_data = np.array(ids_data)
        return image_data, tag_data, ids_data

    def run_model(
        self,
        mode="train",
        trainning_size=0.8,
        num_epoch=20,
        batch_size=100,
        lr=1e-4,
        **kwargs
    ):
        self.logger.info("Running model, %s" % mode)
        data_length = self.tag_data.shape[0]
        self.logger.info("Data size: %d" % data_length)
        self.model.to(self.device)
        input_data = self.image_data
        tags = torch.from_numpy(self.tag_data).to(torch.int64)
        assert mode in ["train", "predict"]
        if mode == "train":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            indexes = np.arange(data_length)
            np.random.shuffle(indexes)
            train_len = round(data_length * trainning_size)
            train_inds = indexes[:train_len]
            self.logger.info("Training data length: %d" % len(train_inds))
            test_inds = indexes[train_len:]
            # 生成的训练、测试数据供测试使用
            # 取训练集的1/10
            train_inds_select = train_inds[::40]
            test_inds_select = test_inds[::10]
            train_data = torch.from_numpy(input_data[train_inds_select])
            self.logger.info("Sample length in training set: %d" % train_data.shape[0])
            train_tags = tags[train_inds_select]

            test_data = torch.from_numpy(input_data[test_inds_select])
            self.logger.info("Sample length in testing set: %d" % test_data.shape[0])
            test_tags = tags[test_inds_select]
            n_batch = math.ceil(len(train_inds) / batch_size)
            self.logger.debug("Batch number: %d" % n_batch)
            best_micro_f1 = 0
            best_epoch = 0
            for epoch in range(num_epoch):

                # if epoch % 10 == 0:
                #     self.logger.info('Epoch number: %d' % epoch)

                if epoch % kwargs["print_interval"] == 0:
                    with torch.no_grad():
                        cpc_pred_train, train_loss = self.model(
                            train_data.to(device=self.device, dtype=torch.float32),
                            train_tags.to(self.device),
                        )
                        cpc_pred_test, test_loss = self.model(
                            test_data.to(device=self.device, dtype=torch.float32),
                            test_tags.to(self.device),
                        )
                        cpc_pred_worst = (
                            cpc_pred_test.cpu().detach().numpy()[:, 0].flatten()
                        )
                        top10 = np.array(cpc_pred_worst).argsort()[::-1][0:10]
                        self.logger.info("Worst Top 10: {}".format(self.ids[top10]))
                        cpc_pred_best = (
                            cpc_pred_test.cpu().detach().numpy()[:, -1].flatten()
                        )
                        top10 = np.array(cpc_pred_best).argsort()[::-1][0:10]
                        self.logger.info("Best Top 10: {}".format(self.ids[top10]))
                        cpc_pred_train = np.argmax(
                            cpc_pred_train.cpu().detach(), axis=1
                        )
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
                        self.logger.info(
                            "Best Micro-F1: %.6lf, epoch %d"
                            % (best_micro_f1, best_epoch)
                        )

                for i in range(n_batch):
                    start = i * batch_size
                    # 别忘了这里用了sigmoid归一化
                    data_inds = train_inds[start : start + batch_size]
                    data = torch.from_numpy(input_data[data_inds]).to(
                        device=self.device, dtype=torch.float32
                    )
                    _tags = tags[data_inds].to(self.device)
                    cpc_pred, loss = self.model(
                        data, _tags
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
            n_batch = self.data_len // self.batch_size
            for i in range(n_batch):
                start = i * self.batch_size
                # 别忘了这里用了sigmoid归一化
                cpc_pred = self.model(
                    input_data[:, start : start + self.batch_size]
                )  # text_hashCodes是一个32-dim文本特征


def main():
    video_cover_parser = cover_parser(
        "../Data/image_data.npy",
        "../Data/kuaishou_data_0421.csv",
        log_level=logging.INFO,
    )
    video_cover_parser.run_model(
        num_epoch=100, batch_size=50, lr=1e-3, print_interval=1
    )


if __name__ == "__main__":
    main()
