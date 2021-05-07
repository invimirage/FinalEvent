#!/usr/bin/env python
# encoding: utf-8
"""
@author: zhangruifeng
@contact: zrf1999@pku.edu.cn
@file: utils.py
@time: 2021/4/12 15:46
@github: local 16351726fa15c85f565b7d5fecdf320ea67a72ef
"""
import pandas as pd
import numpy as np
import logging
import config
import os

def bin_tags(tags, binnum):
    # 将数据映射到所需数量的分位数
    tags_binned = pd.qcut(tags, binnum, labels=False)
    # 按照指定的数值分桶
    # tags_binned = pd.cut(tags, [0, 0.0032, 1], labels=False)
    # 计算指定分位数点的数据
    large_counts_series = pd.Series(tags)
    cut_points = large_counts_series.quantile(np.linspace(0, 1, binnum + 1))
    return tags_binned, cut_points

def init_logger(log_level, name=os.path.split(__file__)[-1]):
    # logging.basicConfig(
    #     format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S"
    # )
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt="%d-%b-%y %H:%M:%S")
    logger = logging.getLogger(name)

    log_dir = config.log_dir
    log_file = 'log_%s.txt' % name
    log_path = os.path.join(log_dir, log_file)
    # 2、创建一个handler，用于写入日志文件
    fh = logging.FileHandler(log_path)
    fh.setLevel(log_level)

    # 再创建一个handler，用于输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(log_level)

    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger