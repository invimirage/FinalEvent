import logging
import pandas as pd
import numpy as np
import json
import re
from matplotlib import pyplot as plt
import torch

data_file = pd.read_csv('Data/kuaishou_data_es.csv')
print(data_file.columns)
c = []
for i, id in zip(data_file['advids'], data_file['id']):
    try:
        c.append(len(i.split(',')))
        if c[-1] > 1:
            print(id)
    except:
        print(id)
print(np.mean(c), np.max(c))

# a = 10
# print(round(a/3))
# a = np.random.random((2,  3, 4))
# print(a)
# print(np.reshape(a, (-1, 4)))
# train_inds = [1, 2, 3]
# separate_points = np.array([0, 10, 30, 100, 160, 900])
# input_data = np.random.random((900, 10))
# map_to_training = separate_points[train_inds], separate_points[np.array(train_inds)+1]
# print(map_to_training)
# train_data = input_data[np.concatenate([np.arange(sta, end) for sta, end in zip(*map_to_training)])]
# print(train_data)
# a = torch.tensor([[1.0, 2.0, 3.0]])
# b = torch.tensor([[1.0, 2.0, 3.0]])
# c = [a, b]
# d = torch.cat(c)
# f = torch.mean(d, dim=1, keepdim=True)
# print(f)
# b = torch.nn.Softmax(1)
# print(b(torch.tensor(a, dtype=torch.float32)))
# ta = torch.tensor(a)
# tb = torch.tensor(b)
# print(ta*tb)

# data = pd.read_csv("Data/kuaishou_data_es.csv")
# text_data = data["full_texts"]
# raw_data = data["audio_text"]
# ids = data["id"]
# seperated_len = []
# seps = []
# sum = 0
# for text, raw, id in zip(text_data, raw_data, ids):
#     text = re.split("[，。,.]", text)
#     sentences = []
#     raw = json.loads(raw)
#     words_with_time = []
#     for res_detail in raw["Response"]["Data"]["ResultDetail"]:
#         words_with_time.extend(res_detail["Words"])
#     word_num = len(words_with_time)
#     sentence = ""
#     for i in range(word_num):
#         sta = words_with_time[i]["OffsetStartMs"]
#         end = words_with_time[i]["OffsetEndMs"]
#         word = words_with_time[i]["Word"]
#         if i < word_num - 1:
#             next_sta = words_with_time[i + 1]["OffsetStartMs"]
#         else:
#             next_sta = 0
#         sentence += word
#         # 考虑分句
#         if int(next_sta) - int(end) > 20 or word in ["，", "。"]:
#             # 必分，查看长度
#             if i == word_num - 1 or word == "。":
#                 # 长度太短，嫩加就加到上一句
#                 if len(sentence) <= 10 and len(sentences) > 0:
#                     sentences[-1] += sentence
#                 else:
#                     sentences.append(sentence)
#                 sentence = ""
#
#             # 不是必分，长度够了才分
#             if len(sentence) > 10:
#                 sentences.append(sentence)
#                 sentence = ""
#
#     for sep in sentences:
#         if len(sep) > 50:
#             sum += 1
#             # print(sep, id)
#             # print(raw)
#         seperated_len.append(len(sep))
#     seps.append(len(sentences))
#     if len(sentences) == 0:
#         print(id, text, raw)
#     # if len(text) > 50:
#     #     print(text)
# seperated_len = np.array(seperated_len)
# seps = np.array(seps)
# print(sum)
# print(seperated_len.min(), seperated_len.mean(), seperated_len.max())
# print(seps.min(), seps.mean(), seps.max())
# fig, subs = plt.subplots(2, 1)
# subs[0].hist(seps, bins=10)
#
# subs[1].hist(seperated_len, bins=10)
#
# plt.show()
# data = pd.read_csv('Data/kuaishou_data_es.csv')
# embed_data = data['audio_text']
# def ddict(d):
#     if not isinstance(d, dict) and not isinstance(d, list):
#         print(d)
#         return
#     if isinstance(d, dict):
#         for k in d:
#             print(k)
#             ddict(d[k])
#     else:
#         for k in d:
#             ddict(k)
#
# for i in embed_data[0:1]:
#     ddict(json.loads(i))
# df_test = pd.DataFrame(columns=['a'])
# test_arr = np.array([i for i in range(10000)])
# df_test['a'] = [test_arr.tolist()]
# df_test.to_csv('test.csv')
# df_2 = pd.read_csv('test.csv')
# print((df_2['a'][0]))
# print(json.loads(df_2['a'][0]))
