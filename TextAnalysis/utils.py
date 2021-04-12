#!/usr/bin/env python
# encoding: utf-8
'''
@author: zhangruifeng
@contact: zrf1999@pku.edu.cn
@file: utils.py
@time: 2021/4/12 15:46
@github: local 16351726fa15c85f565b7d5fecdf320ea67a72ef
'''
import pandas as pd
import numpy as np

def bin_tags(tags, binnum):
    # 将数据映射到所需数量的分位数
    tags_binned = pd.qcut(tags, binnum, labels=False)
    # 计算指定分位数点的数据
    large_counts_series = pd.Series(tags)
    cut_points = large_counts_series.quantile(np.linspace(0, 1, binnum))
    return tags_binned, cut_points
