#!/usr/bin/env python
# encoding: utf-8
'''
@author: zhangruifeng
@contact: zrf1999@pku.edu.cn
@file: video_scorer.py
@time: 2021/5/9 13:38
@github: local 16351726fa15c85f565b7d5fecdf320ea67a72ef
'''

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
from utils import *
import time

import shutil


class VideoScorer:
    def __init__(self, **kwargs):
        self.logger = kwargs['logger']
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"
        self.logger.info("Device: %s" % self.device)
        self.model = kwargs["model"]
        self.model.to(self.device)
        self.best_f1 = kwargs['f1']
        # print(pictures[0:10])

    def set_tag(self, tag_data):
        self.tag= torch.tensor(tag_data)
        self.data_len = self.tag.shape[0]
        self.logger.info("Data length: %d" % self.data_len)

    # convert videos to numpy data
    def extract_frames(self, video_folder, target_file_folder, force=False):
        video_files = list(filter(lambda x: x.endswith('.mp4') and not x.endswith('_origin.mp4'), os.listdir(video_folder)))
        if not force:
            already_extracted = list(map(lambda x: x.split('.')[0], os.listdir(target_file_folder)))
            video_files = [video_file for video_file in video_files if video_file not in already_extracted]
        else:
            shutil.rmtree(target_file_folder)
            os.makedirs(target_file_folder)
        np.random.shuffle(video_files)
        video_framerate = config.video_frame_rate
        target_framerate = self.model.hyperparams["frame_rate"]
        self.logger.info(f"Original frame rate {video_framerate}, target frame rate {target_framerate}")
        img_interval = video_framerate // target_framerate
        for i, video_file in enumerate(video_files):
            if i % 100 == 0:
                self.logger.info("Extracting frames, at %d/%d" % (i, len(video_files)))
            id = video_file.split('.')[0]
            frames = []
            video_path = os.path.join(video_folder, video_file)
            vc = cv2.VideoCapture(video_path)
            frame_count = 0
            while True:
                rval, frame = vc.read()
                if not rval:
                    break
                if frame_count % img_interval == 0:
                    frame = cv2.resize(frame, (100, 100), interpolation=cv2.INTER_LANCZOS4)
                    frame = np.transpose(frame, (2, 0, 1))
                    frames.append(frame)
                    self.logger.debug(frame.shape)
                frame_count += 1
            frames = np.array(frames)
            if frames.shape[0] == 0:
                os.remove(video_path)
            else:
                filename = f"{id}.npy"
                np.save(os.path.join(target_file_folder, filename), frames)
            vc.release()

    def video_checker(self, video_folder):
        video_files = os.listdir(video_folder)
        filter_func = lambda x: x.endswith('.mp4') and not x.endswith('_origin.mp4')
        video_files = list(filter(filter_func, video_files))
        valid_video_files = []
        for video_file in video_files:
            video_path = os.path.join(video_folder, video_file)
            vc = cv2.VideoCapture(video_path)
            rval, _ = vc.read()
            if rval:
                valid_video_files.append(video_file)
        return valid_video_files

    # 指定batch，加载数据
    def load_video_data(self, file_name, **kwargs):
        assert file_name.endswith('mp4') or file_name.endswith('npy')
        if file_name.endswith('mp4'):
            video_file = file_name
            video_framerate = kwargs['frame_rate']
            target_framerate = kwargs['target_rate']
            img_interval = video_framerate // target_framerate
            frames = []
            video_path = os.path.join(config.video_folder, video_file)
            vc = cv2.VideoCapture(video_path)
            frame_count = 0
            while True:
                rval, frame = vc.read()
                if not rval:
                    break
                if frame_count % img_interval == 0:
                    frame = cv2.resize(frame, (100, 100), interpolation=cv2.INTER_LANCZOS4)
                    frame = np.transpose(frame, (2, 0, 1))
                    frames.append(frame)
                    self.logger.debug(frame.shape)
                frame_count += 1
            if len(frames) == 0:
                print(id, frame_count)
            frames = np.array(frames)
            vc.release()
        else:
            frames = np.load(os.path.join(config.frame_data_folder, file_name))
            print(frames.shape)
        return frames

    def run_model(self, mode="train", **kwargs):
        if self.model.name == "VideoNet":
            self.run_video_net(mode, kwargs)


    def run_video_net(self, mode, kwargs):
        batch_size = kwargs["params"]["batch_size"]
        lr = kwargs["params"]["learning_rate"]
        training_size = kwargs["params"]["training_size"]
        num_epoch = kwargs["params"]["number_epochs"]
        frames = kwargs["frames"]

        # 文本数据是分段的，需要构建模型输入数据，即input和seps
        def feed_model(frame_data, extra_data, tag_data, indexes, requires_grad=True):
            input = []
            lengths = []
            extra = []
            prepare_data_sta = time.perf_counter()
            for data_section_id in indexes:
                data_section = self.load_video_data(frame_data[data_section_id], frame_rate=config.video_frame_rate,
                                                    target_rate=self.model.hyperparams["frame_rate"])
                extra_feat = extra_data[data_section_id]
                extra.append(extra_feat)
                input.append(torch.tensor(data_section, dtype=torch.float32))
                lengths.append(len(data_section))

            prepare_data_end = time.perf_counter()
            _tag_data = tag_data[indexes].to(self.device)
            _extra_data = torch.tensor(extra, dtype=torch.float32).to(self.device)
            run_model_sta = time.perf_counter()
            if not requires_grad:
                with torch.no_grad():
                    pred, loss = self.model(input, _extra_data, tag=_tag_data, image_batch_size=kwargs['img_batch_size'], device=self.device)
            else:
                pred, loss = self.model(input, _extra_data, tag=_tag_data, image_batch_size=kwargs['img_batch_size'], device=self.device)
            run_model_end = time.perf_counter()
            print(run_model_end - run_model_sta, prepare_data_end - prepare_data_sta)
            return pred, loss

        self.logger.info("Running model, %s" % mode)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        extra_feats = kwargs["extra_features"]
        tags = self.tag.to(torch.int64)
        assert mode in ["train", "predict"]
        if mode == "train":
            train_inds, test_inds = sep_train_test(self.data_len, tags, training_size)
            train_inds_sample = train_inds[::int(training_size//(1-training_size))]
            self.logger.info("Training data length: %d" % len(train_inds))

            # 生成的训练、测试数据供测试使用
            best_micro_f1 = 0
            best_epoch = 0
            for epoch in range(num_epoch):
                if epoch % kwargs['print_interval'] == 0:
                    train_batches = build_batch(train_inds_sample, batch_size, False, tags[train_inds_sample])
                    test_batches = build_batch(test_inds, batch_size, False, tags)
                    training_preds = []
                    training_losses = []
                    testing_preds = []
                    testing_losses = []
                    for batch_inds in train_batches:
                        pred, loss = feed_model(frames, extra_feats, tags, batch_inds, requires_grad=False)
                        training_preds.append(pred)
                        training_losses.append(loss.unsqueeze(0))
                    pred_train = torch.cat(training_preds, dim=0)
                    train_loss = torch.mean(torch.cat(training_losses, dim=0))
                    for batch_inds in test_batches:
                        pred, loss = feed_model(frames, extra_feats, tags, batch_inds, requires_grad=False)
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
                            save_the_best(self.model, f1_mean, kwargs["ids"], tags[test_inds], pred_test,
                                          self.logger.name)
                    self.logger.info(
                        "Best Micro-F1: %.6lf, epoch %d" % (best_micro_f1, best_epoch)
                    )
                    if epoch - best_epoch > 10:
                        break

                train_batches = build_batch(train_inds, batch_size, True, tags)
                for batch_inds in train_batches:
                    pred, loss = feed_model(
                        frames, extra_feats, tags, batch_inds
                    )  # text_hashCodes是一个32-dim文本特征
                    optimizer.zero_grad()
                    self.logger.debug(loss)
                    loss.backward()
                    # for name, param in self.model.named_parameters():
                    #     self.logger.debug(param.grad)
                    optimizer.step()

                np.random.shuffle(train_inds)

                for name, param in self.model.named_parameters():
                    if name == "fcs.2.bias":
                        self.logger.debug(name, param)
        else:
            pass

def main(model, logger, kwargs):
    extract_frames = kwargs["extract_frames"]
    data = pd.read_csv(config.raw_data_file)
    tag_col = "tag"
    run_params = model.hyperparams["common"]
    best_f1 = load_model(file_name=logger.name)
    video_scorer = VideoScorer(
        model=model,
        logger=logger,
        f1=best_f1
    )
    if extract_frames:
        if not os.path.exists(config.frame_data_folder):
            os.makedirs(config.frame_data_folder)
        logger.info("Loading video frames")
        video_scorer.extract_frames(config.video_folder, config.frame_data_folder, kwargs["force"])
    #     video_files = os.listdir(config.frame_data_folder)
    # else:
    #     video_files = video_scorer.video_checker(config.video_folder)

    video_files = os.listdir(config.frame_data_folder)
    removed_rows = []
    frame_data = []
    frame_data_dict = {}
    for video_file in video_files:
        id = video_file.split('.')[0]
        frame_data_dict[id] = video_file
    for i, row in data.iterrows():
        if str(row['id']) in frame_data_dict.keys():
            frame_data.append(frame_data_dict[str(row['id'])])
        else:
            removed_rows.append(i)
    data.drop(index=data.index[removed_rows], inplace=True)
    data = data.reset_index()
    text_data = data["separated_text"].apply(lambda text: json.loads(text))
    tag_data = np.array(data[tag_col])
    video_scorer.set_tag(tag_data)
    id = np.array(data['id'])
    video_sources = config.video_url_prefix + np.array(data['file'])
    advid_onehot = []
    one_hot_len = len(config.advids)
    advid_dict = {k: v for v, k in enumerate(config.advids)}
    for i in range(video_scorer.data_len):
        advid = data["advid"][i]
        try:
            idx = advid_dict[str(advid)]
            one_hot = np.eye(one_hot_len, dtype=int)[idx]
        except KeyError:
            one_hot = np.zeros(one_hot_len, dtype=int)
        advid_onehot.append(one_hot)
    video_scorer.run_model(
        mode="train",
        extra_features=advid_onehot,
        ids=id,
        # 文件名 .mp4 or .npy
        frames=frame_data,
        video_sources=video_sources,
        params=run_params,
        print_interval=1,
        text=text_data,
        # 用于image embedding
        img_batch_size=500
    )


if __name__ == "__main__":
    with open(config.parameter_file) as f:
        params = json.load(f)
    logger = init_logger(logging.INFO, name="VideoScorer", write_to_file=True, clean_up=True)
    ## Force 覆盖之前的extraction
    adjust_hyperparams(logger, params, 10, "VideoNet", main, extract_frames=True, force=True)
