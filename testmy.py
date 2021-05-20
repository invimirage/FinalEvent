import logging
import pandas as pd
import numpy as np
import json
import re
from matplotlib import pyplot as plt
import torch
import math
from transformers import BertModel, BertTokenizer, BertConfig
import torch.nn.utils.rnn as rnn
import config
import subprocess
import os
import time
from utils import *
import cv2
from models import *
# a = {1:2}

data = pd.read_csv(config.raw_data_file)
print(data["mean_val"], data["mean_tag"])
# a.update({2:1})
# print(a)
# m = torch.tensor([1, 2, 3])
# a = torch.tensor([1, 2, 0])
# b = (a != 0)
# print(m*b)
# with open("./Hyperparameters.json") as f:
#     hp = json.load(f)
# ps = hp["SeparatedLSTM"]
# lstm = BiLSTMWithAttention(input_length=20, extra_length=1, hyperparams=ps)
# input = torch.zeros([2, 10, 20])
# lstm(input, torch.zeros((2, 1)))
# from datetime import datetime
# a = "2020-01-02"
# b = "2021-01-02"
# a = datetime.strptime(a, "%Y-%m-%d")
# b = datetime.strptime(b, "%Y-%m-%d")
# delta = a - b
# print(delta.days)
# a = torch.tensor([1, 2, 3]).to("cuda:0")
# print(a[2:5])
# a = pd.DataFrame([[2, 3, 4], [2, 1, 1]], columns=['a', 'v', 'b'])
# b = pd.DataFrame([[2, 3,5], [2, 5, 1]], columns=['a', 'v', 'b'])
# print(pd.concat((a, b),axis=0, ignore_index=True))
# videos = os.listdir(config.video_folder)
# count = 0
# for v in videos:
#     vpath = os.path.join(config.video_folder, v)
#     vc = cv2.VideoCapture(vpath)
#     rval, _ = vc.read()
#     if not rval:
#         os.remove(vpath)
# print(count)

# from difflib import SequenceMatcher#导入库
# def similarity(a, b):
#     return SequenceMatcher(lambda x: x in [" ", "，", "。"], a, b).ratio()#引用ratio方法，返回序列相似性的度量
#
# data = pd.read_csv(config.raw_data_file)
# text = data["full_texts"]
# tags = data["cost"]
# ids = data['id']
# total_len = len(text)
# rela_count = [0] * 100
# for i in range(100):
#     t1 = text[i]
#     print(t1)
#     print(tags[i], ids[i])
#     for j in range(i + 1, total_len):
#         t2 = text[j]
#         if similarity(t1, t2) > 0.9:
#             print(t2)
#             print(tags[j], ids[j])
#             rela_count[i] += 1
# print(rela_count)


# def same(text1, text2):

# data = pd.read_csv(config.raw_data_file)
# print(data["upload_time"])
# extra_features = parse_extra_features(data)
# print(extra_features)
# a = np.array([1, 2, 3 ,4])
# print(np.random.choice(a, 2, replace=False))
# res = pd.read_csv("results.csv")
# pred = np.array(res['pred'])
# tag = np.array(res['tag'])
# test = pred < 0.35
# count = np.sum(test)
# tags = tag[test]
# preds = pred[test]
# rate = np.sum(tags==0) / count
# print(count, rate)
# video_local_path = r"P:\PycharmProjects\FinalEvent\FinalEvent\Data\videos_sample\714546.mp4"
# video_target_path = r"P:\PycharmProjects\FinalEvent\FinalEvent\Data\videos_sample\714546_test.mp4"
# trans = r'ffmpeg -y  -i "%s" -r 5 -vf scale=180:-1 -map 0:0 "%s"' % (video_local_path, video_target_path)
# p = subprocess.Popen(
#     trans, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
# )
# out, errs = p.communicate(timeout=10)
# print(out)
# print(out.decode("utf-8"))
# # os.remove(video_local_path)
#
# print('deleted')
# out = p.stdout.read()
# temp = str(out.decode("utf-8"))
# print(temp)
# a = np.array([1, 1, 3])
# print(np.sum(a == 1))
# d = {'1':1}
# train_inds = [1, 2, 3 ,4 ,5]
# trainning_size = 0.8
# print(int(trainning_size // (1 - trainning_size)))
# train_inds_sample = train_inds[::int(trainning_size // (1 - trainning_size))]
# print(train_inds_sample)
# a = [torch.tensor([1, 2]), torch.tensor([1, 2, 3])]
# print(torch.cat(a, dim=0))
# a = [torch.tensor([[1, 2, 3], [3, 4, 5]]), torch.tensor([[4, 5, 6]])]
# a = rnn.pad_sequence(a, batch_first=True).to(torch.float32)
# b = rnn.pack_padded_sequence(a, [2, 1], batch_first=True)
# lstm = torch.nn.LSTM(input_size=3, hidden_size=2, num_layers=4, batch_first=True, dropout=0.1)
# print(a)
# print(b)
# out, (hn, cn) = lstm(b, batch_first=True)
# print(out, hn.shape, cn.shape)
# test_tensor = torch.FloatTensor([[1, 2, 3], [1, 2, 3]])
# linear = torch.nn.Linear(3, 1)
# for param in linear.parameters():
#     print(param)
# print(linear(test_tensor))

# bert_model = BertModel.from_pretrained('../Models/Bert')
# requires_grad = False
# for name, para in bert_model.named_parameters():
#     if name == 'encoder.layer.23.attention.self.query.weight':
#         requires_grad = True
#     para.requires_grad = requires_grad
# for name, param in bert_model.named_parameters():
#     print(name, param.requires_grad)
#     if name == 'encoder.layer.23.output.dense.weight':
#         print(param)
# print([1] * 4)
# tokenizer = BertTokenizer(vocab_file="../Models/Bert/vocab.txt")  # 初始化分词器
# s = "[SEP]"
# print(tokenizer.convert_tokens_to_ids(s))
# sentence1 = '我今天吃了馒头，明天吃啥呢？'
# sentence2 = '我明天想吃玉米'
# res = tokenizer.encode_plus(sentence1, sentence2)
# decoded = tokenizer.decode(res['input_ids']).split(' ')
# for i, j in zip(decoded, res['token_type_ids']):
#     print(i, j)
# input = torch.tensor(res['input_ids']).unsqueeze(0)
# for i in range(len(res['token_type_ids'])):
#     res['token_type_ids'][i] = 0
# tokens = torch.tensor(res['token_type_ids']).unsqueeze(0)
# mask = torch.tensor(res['attention_mask']).unsqueeze(0)
# # res = tokenizer.decode(res)
# # res = ['[CLS]'] + res + ['[SEP]']
# # res = tokenizer.convert_tokens_to_ids(res)
# # print(res)
# output = bert_model(input, tokens, mask)[0]
# linear = torch.nn.Linear(1024, 1)
# out = linear(output)
# print(out.shape)
# print(output[0][0][0])
# print(output[0][0][15])
# print(output[0][0][-1])
#
#
# tokenizer = BertTokenizer(vocab_file='../Models/Bert/vocab.txt')  # 初始化分词器
# sentence1 = '我今天吃了馒头，明天吃啥呢？'
# sentence2 = '我明天想吃玉米'
# res = tokenizer.encode_plus(sentence1, sentence2)
# input = torch.tensor(res['input_ids']).unsqueeze(0)
# tokens = torch.tensor(res['token_type_ids']).unsqueeze(0)
# mask = torch.tensor(res['attention_mask']).unsqueeze(0)
# # res = tokenizer.decode(res)
# # res = ['[CLS]'] + res + ['[SEP]']
# # res = tokenizer.convert_tokens_to_ids(res)
# # print(res)
# output = bert_model(input, tokens, mask)
# print(output[0][0][0])
# print(output[0][0][15])
# print(output[0][0][-1])


# print(np.array([1, 2, 3])[0, 1])
# logger = logging.getLogger('aa')
# logging.debug('aaa')
# logger.setLevel(logging.DEBUG)
# logger.debug('www')
# print(logger.disabled)
# print('aaa')
# # 2.利用拉马努金公式计算π
# # doc_k = 1
# # doc_4k = 1
# # doc_396 = 1
# # doc_396_4 = 396 ** 4
# #
# # def fac(n, m):   #定义阶乘函数
# #     f = 1
# #     for i in range(n-m+1,n+1):
# #         f *= i
# #     return f
# # def c(k):  #定义第k项函数
# #     if k == 0:
# #         return 1103
# #     global doc_4k, doc_k, doc_396
# #     doc_4k = doc_4k*fac(4*k, 4)
# #     doc_k = k*doc_k
# #     doc_396 = doc_396 * doc_396_4
# #     c = (doc_4k*(1103+26390*k))/(doc_k**4*doc_396)
# #
# #     return c
# #
# # s = 0
# #
# # for k in range(0,10000):
# #     s += c(k)
# # f = (8**0.5/9801)*s
# # print(1/f)
#
# import pymysql
# def get_data_from_db():
#     # 打开数据库连接
#     db = pymysql.connect(
#         host="192.168.0.18",
#         user="zhangruifeng",
#         passwd="t0dG18PjAJ8c8EgR",
#         database="adwetec_prod",
#     )
#
#     # 使用 cursor() 方法创建一个游标对象 cursor
#     cursor = db.cursor()
#
#     # 使用 execute()  方法执行 SQL 查询
#     sql = "select {} from adwetec_material_upload \
#        where audio_text != '' and !ISNULL(audio_text)".format(
#         ",".join(config.include_cols)
#     )
#     cursor.execute(sql)
#
#     # 使用 fetchone() 方法获取单条数据.
#     data = cursor.fetchall()
#
#     df = pd.DataFrame(data, columns=config.include_cols)
#
#     # 关闭数据库连接
#     db.close()
#     return df
#
#
# # sep_points = [0, 4, 9, 15, 24, 44]
# # advid_onehot = [[0, 1, 0], [1, 0, 1]]
# # process_mark = np.array([1, 2])
# # process_mark = process_mark.reshape([len(process_mark), -1])
# # print(process_mark)
# # extra_features = np.concatenate([advid_onehot, process_mark], axis=1)
# # print(extra_features)
# # process_mark = np.concatenate([np.arange(1, 1 + sep_points[i+1] - sep_points[i]) / (sep_points[i+1] - sep_points[i]) for i in range(len(sep_points)-1)])
# # print(process_mark)
# # agg_fields = {"bclk": "bclk", "pclk": "pclk", "cost": "cost", "clk": "clk"}
# # a = pd.DataFrame(data=[[1, 2, 3], [2, 3, 4]], columns=['A', 'B', 'C'])
# # b = pd.DataFrame(data=[[1, 2, 3], [1, 3, 4], [2, 1, 2]], columns=['A', 'B', 'C'])
# # b.set_index('A', inplace=True)
# # print(b.index)
# # c = a.join(b, on='A', lsuffix='-a')
# # print(c)
# # data_file = pd.read_csv('Data/kuaishou_data_es.csv')
# # print(data_file.columns)
# # c = []
# # for i, id in zip(data_file['advids'], data_file['id']):
# #     try:
# #         c.append(len(i.split(',')))
# #         if c[-1] > 1:
# #             print(id)
# #     except:
# #         print(id)
# # print(np.mean(c), np.max(c))
#
# # a = 10
# # print(round(a/3))
# # a = np.random.random((2,  3, 4))
# # print(a)
# # print(np.reshape(a, (-1, 4)))
# # train_inds = [1, 2, 3]
# # separate_points = np.array([0, 10, 30, 100, 160, 900])
# # input_data = np.random.random((900, 10))
# # map_to_training = separate_points[train_inds], separate_points[np.array(train_inds)+1]
# # print(map_to_training)
# # train_data = input_data[np.concatenate([np.arange(sta, end) for sta, end in zip(*map_to_training)])]
# # print(train_data)
# # a = torch.tensor([[1.0, 2.0, 3.0]])
# # b = torch.tensor([[1.0, 2.0, 3.0]])
# # c = [a, b]
# # d = torch.cat(c)
# # f = torch.mean(d, dim=1, keepdim=True)
# # print(f)
# # b = torch.nn.Softmax(1)
# # print(b(torch.tensor(a, dtype=torch.float32)))
# # ta = torch.tensor(a)
# # tb = torch.tensor(b)
# # print(ta*tb)
#
# # data = pd.read_csv("Data/kuaishou_data_es.csv")
# # text_data = data["full_texts"]
# # raw_data = data["audio_text"]
# # ids = data["id"]
# # seperated_len = []
# # seps = []
# # sum = 0
# # for text, raw, id in zip(text_data, raw_data, ids):
# #     text = re.split("[，。,.]", text)
# #     sentences = []
# #     raw = json.loads(raw)
# #     words_with_time = []
# #     for res_detail in raw["Response"]["Data"]["ResultDetail"]:
# #         words_with_time.extend(res_detail["Words"])
# #     word_num = len(words_with_time)
# #     sentence = ""
# #     for i in range(word_num):
# #         sta = words_with_time[i]["OffsetStartMs"]
# #         end = words_with_time[i]["OffsetEndMs"]
# #         word = words_with_time[i]["Word"]
# #         if i < word_num - 1:
# #             next_sta = words_with_time[i + 1]["OffsetStartMs"]
# #         else:
# #             next_sta = 0
# #         sentence += word
# #         # 考虑分句
# #         if int(next_sta) - int(end) > 20 or word in ["，", "。"]:
# #             # 必分，查看长度
# #             if i == word_num - 1 or word == "。":
# #                 # 长度太短，嫩加就加到上一句
# #                 if len(sentence) <= 10 and len(sentences) > 0:
# #                     sentences[-1] += sentence
# #                 else:
# #                     sentences.append(sentence)
# #                 sentence = ""
# #
# #             # 不是必分，长度够了才分
# #             if len(sentence) > 10:
# #                 sentences.append(sentence)
# #                 sentence = ""
# #
# #     for sep in sentences:
# #         if len(sep) > 50:
# #             sum += 1
# #             # print(sep, id)
# #             # print(raw)
# #         seperated_len.append(len(sep))
# #     seps.append(len(sentences))
# #     if len(sentences) == 0:
# #         print(id, text, raw)
# #     # if len(text) > 50:
# #     #     print(text)
# # seperated_len = np.array(seperated_len)
# # seps = np.array(seps)
# # print(sum)
# # print(seperated_len.min(), seperated_len.mean(), seperated_len.max())
# # print(seps.min(), seps.mean(), seps.max())
# # fig, subs = plt.subplots(2, 1)
# # subs[0].hist(seps, bins=10)
# #
# # subs[1].hist(seperated_len, bins=10)
# #
# # plt.show()
# # data = pd.read_csv('Data/kuaishou_data_es.csv')
# # embed_data = data['audio_text']
# # def ddict(d):
# #     if not isinstance(d, dict) and not isinstance(d, list):
# #         print(d)
# #         return
# #     if isinstance(d, dict):
# #         for k in d:
# #             print(k)
# #             ddict(d[k])
# #     else:
# #         for k in d:
# #             ddict(k)
# #
# # for i in embed_data[0:1]:
# #     ddict(json.loads(i))
# # df_test = pd.DataFrame(columns=['a'])
# # test_arr = np.array([i for i in range(10000)])
# # df_test['a'] = [test_arr.tolist()]
# # df_test.to_csv('test.csv')
# # df_2 = pd.read_csv('test.csv')
# # print((df_2['a'][0]))
# # print(json.loads(df_2['a'][0]))
#
# #!/usr/bin/env python
# # encoding: utf-8
# '''
# @author: caopeng
# @license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
# @contact: deamoncao100@gmail.com
# @software:XXXX
# @file: vedio-logo.py
# @time: 2020/11/3 18:38
# @desc:
# '''
#
# from __future__ import print_function
# import cv2 as cv
# import numpy as np
#
# class VideoAddLogo(object):
#     """
#     给视频加水印的类
#     Input:
#     video_path: 视频路径
#     logo_path: 水印图片路径，必须是png图片，RGBA通道
#     out_path: 输出视频路径
#     以上路径可以调用VideeoAddLogo.set_paths()修改或初始化
#
#     position: 水印位置，取值有
#         带底色字幕形式：
#         'top'： 上方
#         'bottom'：下方
#         静态图片添加：
#         'tl'：左上
#         'tr'：右上
#         'bl'：左下
#         'br'：右下
#         'center'：屏幕中心
#         'full'：全屏
#         以上均保持水印长宽比
#         'select'：框选屏幕区域（不保持水印长宽比）
#     logo_width: 针对除了'full'和'select'的位置形式，设定水印的宽度，取值为(0,1]，代表与视频宽度的比例
#     blank_width: 针对除了'full'和'select'的位置形式，设定水印留白的宽度，取值为[0,0.5)，代表与视频宽度的比例
#     alpha: 水印透明度，取值为[0,1]
#     以上参数可以调用VideeoAddLogo.set_params()修改或初始化
#     """
#     def __init__(self,
#                  video_path='materials/samplead.mp4',
#                  logo_path='materials/hanzi.png',
#                  out_path='materials/output_test.mp4',
#                  position='top',
#                  logo_width=0.1,
#                  blank_width=0.05,
#                  alpha=0.8
#                  ):
#         self.video_path = video_path
#         self.logo_path = logo_path
#         self.out_path = out_path
#         self.position = position
#         assert self.position in ['top', 'bottom', 'tl', 'tr', 'bl', 'br', 'center', 'full', 'select']
#         # 保持长宽比，决定logo的宽度和视频宽度的比例即可
#         self.blank_width = blank_width
#         assert self.blank_width >= 0 and self.blank_width < 0.5
#         self.logo_width = logo_width
#         assert self.logo_width > 0 and self.logo_width <= 1
#         self.alpha = alpha
#         assert self.alpha >= 0 and self.alpha <= 1
#         self.video_frames = None
#         self.fps = None
#         self.total_frames = 0
#         self.logo = None
#         self.logo_pos_begin = 0
#         self.back_ground_color = None
#         if self.video_path is not None:
#             self.video_import(self.video_path)
#
#     """
#     主方法，用于合成新视频并输出，调用前确保已经加载了视频和logo文件
#     """
#     def main_routine(self):
#         if self.logo is None:
#             if self.logo_path != '':
#                 print('Import logo from {}'.format(self.logo_path))
#                 self.logo_import(self.logo_path)
#             else:
#                 raise Exception("Logo isn't imported")
#         parsed_video_frames = self.add_logo()
#         self.video_output(parsed_video_frames, self.out_path)
#
#     """
#     从video_path加载视频，并将结果设置在类的对应元素上
#     Input:
#     video_path: 视频路径
#     Output:
#     frames: 每一帧。list形式，每个元素为ndarry，shpae为视频分辨率*3（rgb）
#     fps: 帧率
#     """
#     def video_import(self, video_path):
#         reader = cv.VideoCapture(video_path)
#         # 获取视频fps
#         fps = reader.get(cv.CAP_PROP_FPS)
#         # 获取视频总帧数
#         frame_all = reader.get(cv.CAP_PROP_FRAME_COUNT)
#         print("[INFO] 视频FPS: {}".format(fps))
#         print("[INFO] 视频总帧数: {}".format(frame_all))
#         print("[INFO] 视频时长: {}s".format(frame_all / fps))
#         frames = []
#         rval, frame = reader.read()
#         count = 0
#         while rval:
#             frames.append(frame)
#             rval, frame = reader.read()
#             percent = count / frame_all
#             # if count == 100:
#             #     break
#             count += 1
#             if count % 50 == 0:
#                 print("Loading videos... {:.2%} percent".format(percent))
#         reader.release()
#         print('Video loaded')
#         self.total_frames = frame_all
#         self.video_frames = frames
#         self.fps = fps
#         if self.logo_path is not None:
#             # 更新视频一定要更新logo
#             self.logo_import(self.logo_path)
#         return frames, fps
#
#     """
#     加载水印图片并进行预处理
#     Input:
#     logo_path: 水印路径
#     Output:
#     logo: 处理后的水印图片，大小和要放在视频上的大小相同
#     logo_pos：水印图片在视频上的初始位置
#     back_color：针对字幕形式的水印的背景色（黑或白），值为[0, 0, 0]或[255, 255, 255]
#     """
#     def logo_import(self, logo_path):
#         if self.video_frames is None:
#             raise Exception("Must import videos before logo")
#         logo, logo_pos, back_color = self.parse_logo(logo_path, self.video_frames[0])
#         self.logo, self.logo_pos_begin, self.back_ground_color = logo, logo_pos, back_color
#         return logo, logo_pos, back_color
#
#     """
#     水印照片预处理
#     Input:
#     frame: 视频中的某一帧，主要使用其shape数值
#     其余输入输出和logo_import相同
#     """
#     def parse_logo(self, logo_path, frame):
#         logo = cv.imread(logo_path, -1)
#         if self.position in ['top', 'bottom']:
#             # 宽度
#             logo_second_dim = int(self.logo_width * frame.shape[0])
#             # 长度
#             logo_first_dim = logo_second_dim * logo.shape[1] // logo.shape[0]
#             logo = cv.resize(logo, (logo_first_dim, logo_second_dim))
#             logo = np.array(logo)
#             alpha = logo[:, :, -1, np.newaxis]
#             self._alpha = np.zeros_like(alpha, dtype='uint8')
#             content_pos = np.nonzero(alpha)
#             self._content_pos = content_pos
#             word_rgb = []
#             logo = logo[:, :, 0:3]
#             for x, y in zip(content_pos[0], content_pos[1]):
#                 word_rgb.append(logo[x][y])
#                 self._alpha[x][y][0] = 1
#             # logo = logo * alpha
#             mean_color = np.mean(word_rgb, axis=0)
#             # print(mean_color)
#             black = [0, 0, 0]
#             white = [255, 255, 255]
#             dist_black = np.linalg.norm(mean_color - black)
#             dist_white = np.linalg.norm(mean_color - white)
#             if dist_black > dist_white:
#                 background_color = black
#             else:
#                 background_color = white
#             if self.position == 'top':
#                 pos_begin = int(self.blank_width * frame.shape[0])
#             else:
#                 pos_begin = frame.shape[0] - logo_second_dim - int(self.blank_width * frame.shape[0])
#             return logo, pos_begin, background_color
#
#         elif self.position == 'select':
#             area = self.select_area(frame, logo)
#             x, y, w, h = area
#             pos_begin = (y, x)
#             logo_first_dim, logo_second_dim = w, h
#
#         else:
#             blank_width = int(self.blank_width * frame.shape[0])
#             # 宽度
#             logo_second_dim = int(self.logo_width * frame.shape[0])
#             # 长度
#             logo_first_dim = logo_second_dim * logo.shape[1] // logo.shape[0]
#             if self.position == 'bl':
#                 pos_begin = (frame.shape[0] - logo_second_dim - blank_width, blank_width)
#             elif self.position == 'br':
#                 pos_begin = (frame.shape[0] - logo_second_dim - blank_width, frame.shape[1] - logo_first_dim - blank_width)
#             elif self.position == 'tl':
#                 pos_begin = (blank_width, blank_width)
#             elif self.position == 'tr':
#                 pos_begin = (blank_width, frame.shape[1] - logo_first_dim - blank_width)
#             elif self.position == 'full':
#                 # 宽度
#                 logo_second_dim = frame.shape[0]
#                 # 长度
#                 logo_first_dim = logo_second_dim * logo.shape[1] // logo.shape[0]
#                 if logo_first_dim > frame.shape[1]:
#                     rate = frame.shape[1] / logo_first_dim
#                     logo_second_dim = int(logo_second_dim * rate)
#                     logo_first_dim = frame.shape[1]
#                 pos_begin = (int(0.5 * (frame.shape[0] - logo_second_dim)), int(0.5 * (frame.shape[1] - logo_first_dim)))
#             else:
#                 pos_begin = (int(0.5 * (frame.shape[0] - logo_second_dim)), int(0.5 * (frame.shape[1] - logo_first_dim)))
#
#         logo = cv.resize(logo, (logo_first_dim, logo_second_dim))
#         logo = np.array(logo)
#         alpha = logo[:, :, -1, np.newaxis]
#         logo = logo[:, :, 0:3]
#         self._alpha = np.zeros_like(alpha, dtype='uint8')
#         content_pos = np.nonzero(alpha)
#         self._content_pos = content_pos
#         for x, y in zip(content_pos[0], content_pos[1]):
#             self._alpha[x][y][0] = 1
#         background_color = None
#         return logo, pos_begin, background_color
#
#     """
#     给视频的每一帧加上水印
#     Output: 加完水印的每一帧
#     """
#     def add_logo(self):
#         print('Parsing')
#         parsed_frames = []
#         frame_shape = self.video_frames[0].shape
#         logo_width = self.logo.shape[0]
#         logo_len = self.logo.shape[1]
#         if self.back_ground_color is not None:
#             background_banner = self.make_banner([logo_width, frame_shape[1]], self.back_ground_color)
#         for i, origin_frame in enumerate(self.video_frames):
#             percent = i / self.total_frames
#             frame = origin_frame.copy()
#             if self.back_ground_color is not None:
#                 logo_begin = max(0, int(percent * (logo_len + frame_shape[1]) - frame_shape[1]))
#                 logo_end = min(logo_len, int(percent * (logo_len + frame_shape[1])))
#                 screen_area_begin = max(0, int(frame_shape[1] - (frame_shape[1] + logo_len) * percent))
#                 screen_area_end = screen_area_begin + logo_end - logo_begin
#                 x1, x2, y1, y2 = self.logo_pos_begin, self.logo_pos_begin + logo_width, screen_area_begin, screen_area_end
#                 frame[x1:x2] = cv.addWeighted(frame[x1:x2], 0.5, background_banner, 0.5, 0.0)
#                 frame = frame.astype('float64')
#                 frame[x1:x2, y1:y2] += self.logo[:, logo_begin:logo_end] * self.alpha - self.alpha * (self._alpha[:, logo_begin:logo_end] * frame[x1:x2, y1:y2])
#                 frame = frame.astype('uint8')
#                 # for x, y in zip(self._content_pos[0], self._content_pos[1]):
#                 #     if y1 + y < y2 and y >= logo_begin:
#                 #         frame[x1 + x][y1 + y - logo_begin] = self.logo[x][y] * self.alpha + frame[x1 + x][y1 + y - logo_begin] * (1 - self.alpha)
#             else:
#                 x1, y1 = self.logo_pos_begin
#                 x2, y2 = x1 + logo_width, y1 + logo_len
#                 # print(x1, x2, y1, y2, self.logo_pos_begin)
#                 frame = frame.astype('float64')
#                 frame[x1:x2, y1:y2] += self.logo * self.alpha - self.alpha * (self._alpha * frame[x1:x2, y1:y2])
#                 frame = frame.astype('uint8')
#                 # frame[x1:x2, y1:y2] = cv.addWeighted(frame[x1:x2, y1:y2], 1.0, self.logo, self.alpha, 0.0)
#             # frame[x1:x2, y1:y2] = cv.addWeighted(frame[x1:x2, y1:y2], 0.0, self.logo[:, l'ogo_begin:logo_end], 1.0, 0.0)
#             parsed_frames.append(frame)
#             if i % 50 == 0:
#                 print("Adding logo... {:.2%} percent".format(i / len(self.video_frames)))
#         return parsed_frames
#
#     """
#     输出处理好的视频到文件
#     Input:
#     frames: 处理好的视频帧
#     out_path: 输出文件路径
#     """
#     def video_output(self, frames, out_path):
#         print('Writing output')
#         fourcc = cv.VideoWriter_fourcc(*"mp4v")
#         writer = cv.VideoWriter(out_path, fourcc, self.fps, frames[0].shape[-2:-4:-1], True)
#         for i, frame in enumerate(frames):
#             writer.write(frame)
#             if i % 50 == 0:
#                 print("Writing output... {:.2%} percent".format(i / len(frames)))
#         writer.release()
#
#     """
#     设定文件路径，当然手动调用import和output函数就不需要设定路径了
#     设定路径后会自动根据情况重新加载video和logo，如果不希望重新加载
#     输入输出见__init__中的说明
#     """
#     def set_paths(self,
#                   video_path=None,
#                   logo_path='materials/hanzi.png',
#                   out_path=None
#                   ):
#         if video_path is not None:
#             self.video_path = video_path
#             # 这是避免video_import中调用logo_import加载过去的logo，因为要重新加载logo
#             if logo_path is not None:
#                 self.logo_path = None
#             self.video_import(video_path)
#         if logo_path is not None:
#             self.logo_path = logo_path
#             self.logo_import(logo_path)
#         if out_path is not None:
#             self.out_path = out_path
#
#     """
#     设定水印参数
#     输入输出见__init__中的说明
#     """
#     def set_params(self,
#                    position=None,
#                    logo_width=None,
#                    blank_width=None,
#                    alpha=None
#                    ):
#         if position is not None:
#             assert position in ['top', 'bottom', 'tl', 'tr', 'bl', 'br', 'center', 'full', 'select']
#             self.position = position
#         if logo_width is not None:
#             assert logo_width > 0 and self.logo_width <= 1
#             self.logo_width = logo_width
#         if blank_width is not None:
#             assert blank_width >= 0 and self.blank_width < 0.5
#             self.blank_width = blank_width
#         if alpha is not None:
#             assert alpha >= 0 and self.alpha <= 1
#             self.alpha = alpha
#         # 更改参数后必须重新加载logo
#         if self.logo_path is not None and self.video_frames is not None:
#             self.logo_import(self.logo_path)
#
#     def make_banner(self, dims, color):
#         color = np.array(color, dtype='uint8')[np.newaxis, :]
#         pure_color_banner = np.repeat(color, dims[0] * dims[1], axis=0)
#         # print(pure_color_banner)
#         pure_color_banner = np.reshape(pure_color_banner, (dims[0], dims[1], color.shape[-1]))
#         # print(pure_color_banner)
#         return pure_color_banner
#
#     def select_area(self, frame, logo):
#         logo = logo[:, :, 0:3]
#         while True:
#             sample_frame = frame.copy()
#             length2width = sample_frame.shape[0] / sample_frame.shape[1]
#             roi = cv.selectROI(windowName="roi", img=sample_frame, showCrosshair=False, fromCenter=False)
#             cv.destroyWindow('roi')
#             x, y, w, h = roi
#             logo = cv.resize(logo, (w, h))
#             sample_frame[y:y + h, x:x + w] = cv.addWeighted(sample_frame[y:y + h, x:x + w], 1.0, logo, self.alpha, 0.0)
#             # sample_frame[y:y + h, x:x + w] = self.blend_logo(0.9, sample_frame[y:y + h, x:x + w], 'materials/logo2.png')
#             cv.resizeWindow('frame', int(length2width * 500), 500)
#             cv.imshow('frame', sample_frame)
#             cv.waitKey(0)
#             ok = input('这样子满意吗？0不满意，其他输入满意')
#             cv.destroyAllWindows()
#             if ok != '0':
#                 break
#         return roi
#
# if __name__=="__main__":
#     # 构建一个实例，直接运行
#     parser = VideoAddLogo(position='center', alpha=0.6, logo_path='materials/logo2.png')
#     parser.main_routine()
#
#     # 在运行中更换参数，但是不更换视频
#     # for pos in ['top', 'bottom', 'tl', 'tr', 'bl', 'br', 'center', 'full', 'select']:
#     #     for logo in ['hanzi', 'logo2', 'black']:
#     #         parser.set_paths(logo_path='materials/{}.png'.format(logo), out_path='materials/{}_{}.mp4'.format(logo, 'select'))
#     #         parser.set_params(position='select')
#     #         parser.main_routine()
#
#     # 手动进行处理
#     # parser = VideoAddLogo(video_path=None)
#     # parser.set_params(position='top', blank_width=0)
#     # parser.video_import(video_path='materials/samplead.mp4')
#     # parser.logo_import(logo_path='materials/black.png')
#     # parser.main_routine()
#
#
#
