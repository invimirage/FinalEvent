#!/usr/bin/env python
# encoding: utf-8
'''
@author: zhangruifeng
@contact: zrf1999@pku.edu.cn
@file: test.py
@time: 2021/4/13 11:22
@github: local 16351726fa15c85f565b7d5fecdf320ea67a72ef
'''
from efficientnet_pytorch import EfficientNet
import numpy as np
model = EfficientNet.from_name('efficientnet-b4')

# ... image preprocessing as in the classification example ...
img = np.random.random([1, 3, 223, 223])
print(img.shape) # torch.Size([1, 3, 224, 224])

features = model.extract_features(img)

print(features.shape) # torch.Size([1, 1280, 7, 7])