#!/usr/bin/env python
# encoding: utf-8
"""
@author: zhangruifeng
@contact: zrf1999@pku.edu.cn
@file: video_scorer.py
@time: 2021/5/9 13:38
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
from utils import *
import time

import shutil


class VideoScorer:
    def __init__(self, **kwargs):
        self.logger = kwargs["logger"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"
        self.logger.info("Device: %s" % self.device)
        self.model = kwargs["model"]
        self.model.to(self.device)
        self.best_f1 = kwargs["f1"]
        # print(pictures[0:10])

    def set_tag(self, tag_data):
        self.tag = torch.tensor(tag_data)
        self.data_len = self.tag.shape[0]
        self.logger.info("Data length: %d" % self.data_len)

    # convert videos to numpy data
    def extract_frames(self, video_folder, target_file_folder, force=False):
        video_files = list(
            filter(
                lambda x: x.endswith(".mp4") and not x.endswith("_origin.mp4"),
                os.listdir(video_folder),
            )
        )
        if not force:
            already_extracted = list(
                map(lambda x: x.split(".")[0] + ".mp4", os.listdir(target_file_folder))
            )
            video_files = [
                video_file
                for video_file in video_files
                if video_file not in already_extracted
            ]
        else:
            input("Empty?")
            shutil.rmtree(target_file_folder)
            os.makedirs(target_file_folder)
        np.random.shuffle(video_files)
        video_framerate = config.video_frame_rate
        try:
            target_framerate = self.model.hyperparams["frame_rate"]
        except KeyError:
            target_framerate = self.model.hyperparams["video"]["frame_rate"]
        self.logger.info(
            f"Original frame rate {video_framerate}, target frame rate {target_framerate}"
        )
        img_interval = video_framerate // target_framerate
        for i, video_file in enumerate(video_files):
            if i % 100 == 0:
                self.logger.info("Extracting frames, at %d/%d" % (i, len(video_files)))
            id = video_file.split(".")[0]
            frames = []
            video_path = os.path.join(video_folder, video_file)
            vc = cv2.VideoCapture(video_path)
            frame_count = 0
            while True:
                rval, frame = vc.read()
                if not rval:
                    break
                if frame_count % img_interval == 0:
                    frame = cv2.resize(
                        frame, (100, 100), interpolation=cv2.INTER_LANCZOS4
                    )
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
        filter_func = lambda x: x.endswith(".mp4") and not x.endswith("_origin.mp4")
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
        assert file_name.endswith("mp4") or file_name.endswith("npy")
        if file_name.endswith("mp4"):
            video_file = file_name
            video_framerate = kwargs["frame_rate"]
            target_framerate = kwargs["target_rate"]
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
                    frame = cv2.resize(
                        frame, (100, 100), interpolation=cv2.INTER_LANCZOS4
                    )
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
        return frames

    def run_model(self, mode="train", **kwargs):
        if self.model.name == "VideoNet":
            self.run_video_net(mode, kwargs)
        if self.model.name == "VideoNetEmbed":
            self.run_video_net_with_embed(mode, kwargs)
        if self.model.name == "JointNet":
            self.run_video_text_net(mode, kwargs)

    # video_input_folder包含所有video的numpy数组，使用id来确定顺序
    def image_embedding(self, video_input_folder):
        self.logger.info("Embedding videos")
        ec_model = EfficientNet.from_name("efficientnet-b4")
        net_weight = torch.load(
            os.path.join(config.efficient_path, "efficientnet-b4-6ed6700e.pth")
        )
        video_files = os.listdir(video_input_folder)
        ec_model.load_state_dict(net_weight)
        ec_model.to(self.device)
        ec_model.eval()
        frames_data = {}
        if not os.path.exists(config.img_embed_folder):
            os.makedirs(config.img_embed_folder)
        for i, video_file in enumerate(video_files):
            if i % 100 == 0:
                self.logger.info('Process: %d/%d' % (i, len(video_files)))
            if i > 0 and i % 100 == 0:
                np.savez(os.path.join(config.img_embed_folder, f"{i}.npz"), frames_data)
                frames_data = {}
            id = video_file.split('.')[0]
            video_file_path = os.path.join(video_input_folder, video_file)
            frames = self.load_video_data(video_file_path,  frame_rate=config.video_frame_rate, target_rate=self.model.hyperparams["frame_rate"])
            frames_tensor = torch.from_numpy(frames).to(device=self.device, dtype=torch.float32)
            with torch.no_grad():
                frame_embeds = ec_model._avg_pooling(ec_model.extract_features(frames_tensor)).flatten(start_dim=1)
                # print(frame_embeds.shape)
            frames_data[id] = frame_embeds.cpu().numpy()
        return frames_data

    def run_video_net_with_embed(self, mode, kwargs):
        batch_size = kwargs["params"]["batch_size"]
        lr = kwargs["params"]["learning_rate"]
        training_size = kwargs["params"]["training_size"]
        num_epoch = kwargs["params"]["number_epochs"]
        frames = kwargs["frames"]

        # 文本数据是分段的，需要构建模型输入数据，即input和seps
        def feed_model(frame_data, extra_data, tag_data, indexes, requires_grad=True):
            input = []
            extra = []
            for data_section_id in indexes:
                data_section = frame_data[data_section_id]
                extra_feat = extra_data[data_section_id]
                extra.append(extra_feat)
                input.append(torch.tensor(data_section, device=self.device, dtype=torch.float32))

            _tag_data = tag_data[indexes].to(self.device)
            _extra_data = torch.tensor(extra, dtype=torch.float32).to(self.device)
            if not requires_grad:
                with torch.no_grad():
                    pred, loss = self.model(
                        input,
                        _extra_data,
                        tag=_tag_data
                    )
            else:
                pred, loss = self.model(
                    input,
                    _extra_data,
                    tag=_tag_data
                )
            return pred, loss

        self.logger.info("Running model, %s" % mode)
        extra_feats = kwargs["extra_features"]
        tags = self.tag.to(torch.int64)

        assert mode in ["train", "predict"]
        if mode == "train":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            train_inds, test_inds = sep_train_test(self.data_len, tags, training_size)
            train_inds_sample = train_inds[:: int(training_size // (1 - training_size))]
            self.logger.info("Training data length: %d" % len(train_inds))

            # 生成的训练、测试数据供测试使用
            best_micro_f1 = 0
            best_epoch = 0
            for epoch in range(num_epoch):
                if epoch % kwargs["print_interval"] == 0:
                    train_batches = build_batch(
                        train_inds_sample, batch_size, False, tags[train_inds_sample]
                    )
                    test_batches = build_batch(test_inds, batch_size, False, tags)
                    training_preds = []
                    training_losses = []
                    testing_preds = []
                    testing_losses = []
                    for batch_inds in train_batches:
                        pred, loss = feed_model(
                            frames, extra_feats, tags, batch_inds, requires_grad=False
                        )
                        training_preds.append(pred)
                        training_losses.append(loss.unsqueeze(0))
                    pred_train = torch.cat(training_preds, dim=0)
                    train_loss = torch.mean(torch.cat(training_losses, dim=0))
                    for batch_inds in test_batches:
                        pred, loss = feed_model(
                            frames, extra_feats, tags, batch_inds, requires_grad=False
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

    def run_video_text_net(self, mode, kwargs):
        batch_size = kwargs["params"]["batch_size"]
        lr = kwargs["params"]["learning_rate"]
        training_size = kwargs["params"]["training_size"]
        num_epoch = kwargs["params"]["number_epochs"]
        frames = kwargs["frames"]
        text_embed_data = kwargs["text_data"]
        random_batching = kwargs["params"]["random"] == 1

        # 文本数据是分段的，需要构建模型输入数据，即input和seps
        def feed_model(frame_data, text_data, extra_data, tag_data, indexes, requires_grad=True):
            text_input = []
            frame_input = []
            lengths = []
            extra = []
            for data_section_id in indexes:
                frame_data_section = frame_data[data_section_id]
                # frame_data_section = self.load_video_data(
                #     frame_data[data_section_id],
                #     frame_rate=config.video_frame_rate,
                #     target_rate=self.model.hyperparams["video"]["frame_rate"],
                # )
                text_data_section = text_data[data_section_id]
                extra_feat = extra_data[data_section_id]
                extra.append(extra_feat)
                text_input.append(torch.tensor(text_data_section, dtype=torch.float32))
                frame_input.append(torch.tensor(frame_data_section, device=self.device, dtype=torch.float32))
                lengths.append(len(text_data_section))
            input_padded = rnn.pad_sequence(text_input, batch_first=True)
            self.logger.debug("padded {}".format(input_padded))
            _input_packed = rnn.pack_padded_sequence(
                input_padded, lengths=lengths, batch_first=True, enforce_sorted=False
            ).to(self.device)
            _tag_data = tag_data[indexes].to(self.device)
            _extra_data = torch.tensor(extra, dtype=torch.float32).to(self.device)
            if not requires_grad:
                with torch.no_grad():
                    pred, loss = self.model(_input_packed, frame_input, _extra_data, _tag_data)
            else:
                pred, loss = self.model(_input_packed, frame_input, _extra_data, _tag_data)
            return pred, loss

        self.logger.info("Running model, %s" % mode)
        extra_feats = kwargs["extra_features"]
        tags = self.tag.to(torch.int64)

        assert mode in ["train", "predict"]
        if mode == "train":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            train_inds, test_inds = sep_train_test(self.data_len, tags, training_size)
            train_inds_sample = train_inds[:: int(training_size // (1 - training_size))]
            self.logger.info("Training data length: %d" % len(train_inds))

            # 生成的训练、测试数据供测试使用
            best_micro_f1 = 0
            best_epoch = 0
            for epoch in range(num_epoch):
                if epoch % kwargs["print_interval"] == 0:
                    train_batches = build_batch(
                        train_inds_sample, batch_size, False, tags[train_inds_sample]
                    )
                    test_batches = build_batch(test_inds, batch_size, False, tags)
                    training_preds = []
                    training_losses = []
                    testing_preds = []
                    testing_losses = []
                    for batch_inds in train_batches:
                        pred, loss = feed_model(
                            frames, text_embed_data, extra_feats, tags, batch_inds, requires_grad=False
                        )
                        training_preds.append(pred)
                        training_losses.append(loss.unsqueeze(0))
                    pred_train = torch.cat(training_preds, dim=0)
                    train_loss = torch.mean(torch.cat(training_losses, dim=0))
                    for batch_inds in test_batches:
                        pred, loss = feed_model(
                            frames, text_embed_data, extra_feats, tags, batch_inds, requires_grad=False
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

                train_batches = build_batch(train_inds, batch_size, random_batching, tags)
                for batch_inds in train_batches:
                    pred, loss = feed_model(
                        frames, text_embed_data, extra_feats, tags, batch_inds
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
                data_section = self.load_video_data(
                    frame_data[data_section_id],
                    frame_rate=config.video_frame_rate,
                    target_rate=self.model.hyperparams["frame_rate"],
                )
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
                    pred, loss = self.model(
                        input,
                        _extra_data,
                        tag=_tag_data,
                        image_batch_size=kwargs["img_batch_size"],
                        device=self.device,
                    )
            else:
                pred, loss = self.model(
                    input,
                    _extra_data,
                    tag=_tag_data,
                    image_batch_size=kwargs["img_batch_size"],
                    device=self.device,
                )
            run_model_end = time.perf_counter()
            # print(run_model_end - run_model_sta, prepare_data_end - prepare_data_sta)
            return pred, loss

        self.logger.info("Running model, %s" % mode)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        extra_feats = kwargs["extra_features"]
        tags = self.tag.to(torch.int64)
        assert mode in ["train", "predict"]
        if mode == "train":
            train_inds, test_inds = sep_train_test(self.data_len, tags, training_size)
            train_inds_sample = train_inds[:: int(training_size // (1 - training_size))]
            self.logger.info("Training data length: %d" % len(train_inds))

            # 生成的训练、测试数据供测试使用
            best_micro_f1 = 0
            best_epoch = 0
            for epoch in range(num_epoch):
                if epoch % kwargs["print_interval"] == 0:
                    train_batches = build_batch(
                        train_inds_sample, batch_size, False, tags[train_inds_sample]
                    )
                    test_batches = build_batch(test_inds, batch_size, False, tags)
                    training_preds = []
                    training_losses = []
                    testing_preds = []
                    testing_losses = []
                    for batch_inds in train_batches:
                        pred, loss = feed_model(
                            frames, extra_feats, tags, batch_inds, requires_grad=False
                        )
                        training_preds.append(pred)
                        training_losses.append(loss.unsqueeze(0))
                    pred_train = torch.cat(training_preds, dim=0)
                    train_loss = torch.mean(torch.cat(training_losses, dim=0))
                    for batch_inds in test_batches:
                        pred, loss = feed_model(
                            frames, extra_feats, tags, batch_inds, requires_grad=False
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
    data = pd.read_csv(config.raw_data_file)
    method_name = logger.name
    best_f1 = load_model(file_name=method_name)
    video_scorer = VideoScorer(model=model, logger=logger, f1=best_f1)
    tag_col = kwargs.get("tag_col", "tag")
    print(tag_col)
    run_params = model.hyperparams["common"]
    print(run_params)
    if model.requires_embed:
        try:
            embed = {}
            get_memory_status("init")
            for embed_file in os.listdir(config.img_embed_folder):
                embed_file_path = os.path.join(config.img_embed_folder, embed_file)
                _embed = np.load(embed_file_path, allow_pickle=True)['arr_0'][()]
                for key in _embed:
                    # 减半
                    embed[key] = _embed[key][::4, :]
                del _embed
                gc.collect()
            get_memory_status("video_embed")
            print(len(embed))
        except:
            embed = video_scorer.image_embedding(config.video_folder)
        frame_data = []
        removed_rows = []
        for i, row in data.iterrows():
            if str(row["id"]) in embed.keys():
                frame_data.append(embed[str(row["id"])])
            else:
                removed_rows.append(i)
        get_memory_status(617)
        del embed
        gc.collect()
        get_memory_status(619)
    else:
        extract_frames = kwargs.get("extract_frames", True)
        if extract_frames:
            if not os.path.exists(config.frame_data_folder):
                os.makedirs(config.frame_data_folder)
            logger.info("Loading video frames")
            video_scorer.extract_frames(
                config.video_folder, config.frame_data_folder, kwargs.get("force", False)
            )
        #     video_files = os.listdir(config.frame_data_folder)
        # else:
        #     video_files = video_scorer.video_checker(config.video_folder)

        video_files = os.listdir(config.frame_data_folder)
        removed_rows = []
        frame_data = []
        frame_data_dict = {}
        for video_file in video_files:
            id = video_file.split(".")[0]
            frame_data_dict[id] = video_file
        for i, row in data.iterrows():
            if str(row["id"]) in frame_data_dict.keys():
                frame_data.append(frame_data_dict[str(row["id"])])
            else:
                removed_rows.append(i)
    data.drop(index=data.index[removed_rows], inplace=True)
    data = data.reset_index()
    text_data = data["separated_text"].apply(lambda text: json.loads(text))
    tag_data = np.array(data[tag_col])
    video_scorer.set_tag(tag_data)
    key = np.array(data["key"])
    video_sources = config.video_url_prefix + np.array(data["file"])
    extra_feats = parse_extra_features(data)
    get_memory_status("text embed start")
    # 需要文本信息
    if model.name == "JointNet":
        requires_embed = model.text_net.requires_embed
        if requires_embed:
            embed_file_path = kwargs.get("text_embed_file", config.embed_data_file + f"_{method_name.split('&')[0]}.npy")
            print(embed_file_path)
            try:
                embed_data = np.load(embed_file_path, allow_pickle=True).tolist()
                get_memory_status("text embed end")
                # embed_data = [embed_data[i] for i in range(len(embed_data)) if i not in removed_indexes]
            except:
                print("Text embedding not found! Do embed first!")
                exit(0)
            video_scorer.run_model(
                mode="train",
                extra_features=extra_feats,
                ids=key,
                # 文件名 .mp4 or .npy
                frames=frame_data,
                video_sources=video_sources,
                params=run_params,
                print_interval=1,
                text=text_data,
                text_data=embed_data,
                # 用于image embedding
                img_batch_size=config.img_batch_size,
            )
    else:
        video_scorer.run_model(
            mode="train",
            extra_features=extra_feats,
            ids=key,
            # 文件名 .mp4 or .npy
            frames=frame_data,
            video_sources=video_sources,
            params=run_params,
            print_interval=1,
            text=text_data,
            # 用于image embedding
            img_batch_size=config.img_batch_size,
        )


if __name__ == "__main__":
    with open(config.parameter_file) as f:
        params = json.load(f)
    logger = init_logger(
        logging.INFO, name="VideoScorer", write_to_file=True, clean_up=True
    )
    ## Force 覆盖之前的extraction
    adjust_hyperparams(
        logger, params, 10, "VideoNet", main, extract_frames=True, force=False
    )
