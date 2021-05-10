#!/usr/bin/env python
# encoding: utf-8
"""
@author: zhangruifeng
@contact: zrf1999@pku.edu.cn
@file: get_videos.py
@time: 2021/3/22 15:16
@github: local 16351726fa15c85f565b7d5fecdf320ea67a72ef
"""
import pandas as pd
import config

data = pd.read_csv(config.raw_data_file)
print(data.columns)
for i in data["file"][:10]:
    print("https://constrain.adwetec.com/material/creative/video/" + i)
