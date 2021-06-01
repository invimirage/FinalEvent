#!/usr/bin/env python
# encoding: utf-8
'''
@author: zhangruifeng
@contact: zrf1999@pku.edu.cn
@file: joint_models.py
@time: 2021/5/29 14:55
@github: local 16351726fa15c85f565b7d5fecdf320ea67a72ef
'''

from TextAnalysis.text_scorer import TextScorer
from VideoAnalysis.video_scorer import VideoScorer
from utils import *
import time
from multiprocessing import Process, Array, Value, Lock
import shutil

class JointScorer:
    def __init__(self, **kwargs):
        self.logger = kwargs["logger"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"
        self.logger.info("Device: %s" % self.device)
        self.model = kwargs["model"]
        self.model.to(self.device)
        self.best_f1 = kwargs["f1"]

    def set_tag(self, tag_data):
        self.tag = torch.tensor(tag_data)
        self.data_len = self.tag.shape[0]
        self.logger.info("Data length: %d" % self.data_len)

