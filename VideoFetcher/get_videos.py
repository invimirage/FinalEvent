#!/usr/bin/env python
# encoding: utf-8
'''
@author: zhangruifeng
@contact: zrf1999@pku.edu.cn
@file: get_videos.py
@time: 2021/3/22 15:16
@github: local 16351726fa15c85f565b7d5fecdf320ea67a72ef
'''
import pandas as pd
video_sources = pd.read_csv('SourceURL/test_data.csv', encoding='utf-8')
print(video_sources.head(10))