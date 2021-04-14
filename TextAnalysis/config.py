#!/usr/bin/env python
# encoding: utf-8
'''
@author: zhangruifeng
@contact: zrf1999@pku.edu.cn
@file: config.py
@time: 2021/4/9 16:19
@github: local 16351726fa15c85f565b7d5fecdf320ea67a72ef
'''

all_cols = ['id', 'name', 'file', 'first_frame', 'cluster', 'file_type', 'status',
       'create_time', 'update_time', 'creator_id', 'duration', 'width',
       'height', 'phash', 'advids', 'ffmpeg_info', 'entities', 'phash2',
       'materialid', 'materialurl', 'direct_id', 'shoot_id', 'script_id',
       'subtitle', 'describe', 'report_unchange', 'report_updatetime',
       'baidu_feed_upload', 'baidu_feed_imp', 'baidu_feed_clk',
       'baidu_feed_cost', 'baidu_feed_begin', 'kuaishou_feed_upload',
       'kuaishou_feed_imp', 'kuaishou_feed_clk', 'kuaishou_feed_cost',
       'kuaishou_feed_begin', 'toutiao_feed_upload', 'toutiao_feed_imp',
       'toutiao_feed_clk', 'toutiao_feed_cost', 'toutiao_feed_begin',
       'tencent_feed_upload', 'tencent_feed_imp', 'tencent_feed_clk',
       'tencent_feed_cost', 'tencent_feed_begin', 'watermask1',
       'watermask1_frist_frame', 'collect_nums', 'sound', 'audio_text']

include_cols = [
        'id', 'name', 'file', 'first_frame', 'duration', 'width',
       'baidu_feed_upload', 'baidu_feed_imp', 'baidu_feed_clk',
       'baidu_feed_cost', 'baidu_feed_begin', 'kuaishou_feed_upload',
       'kuaishou_feed_imp', 'kuaishou_feed_clk', 'kuaishou_feed_cost',
       'kuaishou_feed_begin', 'toutiao_feed_upload', 'toutiao_feed_imp',
       'toutiao_feed_clk', 'toutiao_feed_cost', 'toutiao_feed_begin',
       'tencent_feed_upload', 'tencent_feed_imp', 'tencent_feed_clk',
       'tencent_feed_cost', 'audio_text']
# 素材展现量阈值
threshold = 200

# cpc分桶数量
bin_number = 3

# bert输入维度
bert_batch_size = 1