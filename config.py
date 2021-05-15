#!/usr/bin/env python
# encoding: utf-8
"""
@author: zhangruifeng
@contact: zrf1999@pku.edu.cn
@file: config.py
@time: 2021/4/9 16:19
@github: local 16351726fa15c85f565b7d5fecdf320ea67a72ef
"""
import os


this_path = os.path.realpath(__file__)
this_folder = os.path.split(this_path)[0]
# 需要修改下面的文件夹
data_folder = os.path.join(this_folder, "Data", "kuaishou_data_0514")
# data_folder = os.path.join(this_folder, "Data", "sample")
img_folder = os.path.join(this_folder, "Data", "images")
video_folder = os.path.join(this_folder, "Data", "videos")
raw_data_file = os.path.join(data_folder, "data.csv")
grouped_data_file = os.path.join(data_folder, "grouped.npy")
embed_data_file = os.path.join(data_folder, "vector_embed")
img_embed_file = os.path.join(data_folder, "img_embed.npy")
# frame_data_folder = os.path.join(this_folder, "Data", "frames")
frame_data_folder = r"E:\frames"
video_frame_rate = 5

model_save_path = os.path.join(this_folder, "Models")
bert_path = os.path.join(this_folder, "Models", "Bert")
efficient_path = os.path.join(this_folder, "Models", "efficientnet-b4")
parameter_file = os.path.join(this_folder, "Hyperparameters.json")
log_dir = os.path.join(this_folder, "Logs")

video_url_prefix = "https://constrain.adwetec.com/material/creative/video/"

graph_folder = os.path.join(this_folder, "Graphs")

all_cols = [
    "id",
    "name",
    "file",
    "first_frame",
    "cluster",
    "file_type",
    "status",
    "create_time",
    "update_time",
    "creator_id",
    "duration",
    "width",
    "height",
    "phash",
    "advids",
    "ffmpeg_info",
    "entities",
    "phash2",
    "materialid",
    "materialurl",
    "direct_id",
    "shoot_id",
    "script_id",
    "subtitle",
    "describe",
    "report_unchange",
    "report_updatetime",
    "baidu_feed_upload",
    "baidu_feed_imp",
    "baidu_feed_clk",
    "baidu_feed_cost",
    "baidu_feed_begin",
    "kuaishou_feed_upload",
    "kuaishou_feed_imp",
    "kuaishou_feed_clk",
    "kuaishou_feed_cost",
    "kuaishou_feed_begin",
    "toutiao_feed_upload",
    "toutiao_feed_imp",
    "toutiao_feed_clk",
    "toutiao_feed_cost",
    "toutiao_feed_begin",
    "tencent_feed_upload",
    "tencent_feed_imp",
    "tencent_feed_clk",
    "tencent_feed_cost",
    "tencent_feed_begin",
    "watermask1",
    "watermask1_frist_frame",
    "collect_nums",
    "sound",
    "audio_text",
]

include_cols = [
    "id",
    "name",
    "file",
    "first_frame",
    "duration",
    "advids",
    "width",
    "height",
    "baidu_feed_upload",
    "baidu_feed_imp",
    "baidu_feed_clk",
    "baidu_feed_cost",
    "baidu_feed_begin",
    "kuaishou_feed_upload",
    "kuaishou_feed_imp",
    "kuaishou_feed_clk",
    "kuaishou_feed_cost",
    "kuaishou_feed_begin",
    "toutiao_feed_upload",
    "toutiao_feed_imp",
    "toutiao_feed_clk",
    "toutiao_feed_cost",
    "toutiao_feed_begin",
    "tencent_feed_upload",
    "tencent_feed_imp",
    "tencent_feed_clk",
    "tencent_feed_cost",
    "audio_text",
]
# 素材展现量阈值
threshold = 100

# 广告主素材量阈值
advid_threshold = 10

# cpc分桶数量
bin_number = 2

# bert输入维度
bert_batch_size = 10
max_word_number = 500

# 特殊的广告主
advids = [
    "407",
    "555",
    "1175",
    "929",
    "933",
    "574",
    "932",
    "680",
    "539",
    "1271",
    "1213",
    "537",
    "606",
    "1018",
    "575",
    "703",
    "670",
    "931",
    "543",
    "576",
    "990",
    "569",
    "780",
    "633",
    "673",
    "682",
    "512",
    "496",
    "1202",
    "480",
    "895",
    "1323",
    "618",
    "827",
    "1253",
    "865",
    "658",
    "1185",
    "525",
    "896",
    "1209",
    "547",
    "1280",
    "689",
    "603",
    "473",
    "64",
    "454",
    "283",
    "804",
    "805",
    "690",
    "175",
    "769",
    "596",
    "488",
    "655",
    "578",
    "1216",
    "1104",
]
extra_feat_length = len(advids) + 5
img_batch_size = 1000