#!/usr/bin/env python
# encoding: utf-8
'''
@author: zhangruifeng
@contact: zrf1999@pku.edu.cn
@file: basic_data_handler.py
@time: 2021/5/4 13:24
@github: local 16351726fa15c85f565b7d5fecdf320ea67a72ef
'''

import numpy as np
import pandas as pd
import config
from matplotlib import pyplot as plt
from sklearn import linear_model

class DataHandler:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.data['key'] = list(map(lambda x1, x2: str(x1) + ' ' + str(x2), self.data['id'], self.data['advid']))
        self.gen_tags()

    def gen_tags(self):
        data = self.data
        datalen = len(data)
        imp = np.array(data["clk"])
        beh = np.array(data["bclk"])
        id = np.array(data["id"])

        for i in range(datalen):
            if imp[i] < config.threshold:
                beh[i] = 0
            # 避免分母为0
            if imp[i] == 0:
                imp[i] = 1
        tag_data = beh / imp
        data['tag'] = tag_data
        self.data = data

    def cal_bctr_avg(self):
        data = self.data
        bctr_avg = data.groupby(['advid'], as_index=False)['tag'].agg({'bctr':'mean', 'std': 'std', 'count': 'count'})
        bctr_avg.to_csv('./Data/bctr_avg.csv', index=False)
        plt.hist(bctr_avg['bctr'], bins=10)
        plt.show()

    def relations_bctr_imp(self):
        bctr = np.array(self.data['tag'])[::100]
        imp = np.log(np.array(self.data['clk'])[::100]+1e-5)
        regr = linear_model.LinearRegression()
        regr.fit(imp.reshape(-1, 1), bctr)
        plt.scatter(imp, bctr)
        plt.plot(imp, regr.predict(imp.reshape(-1, 1)), color='red', linewidth=4)
        plt.show()

    def check_data(self, data_id):
        data = self.data
        row_num = list(data['key']).index(data_id)
        print(data[row_num])

if __name__ == '__main__':
    data_handler = DataHandler('./Data/kuaishou_data_0426.csv')
    data_handler.cal_bctr_avg()
    data_handler.relations_bctr_imp()