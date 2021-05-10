#!/usr/bin/env python
# encoding: utf-8
"""
@author: zhangruifeng
@contact: zrf1999@pku.edu.cn
@file: test.py
@time: 2021/4/13 11:22
@github: local 16351726fa15c85f565b7d5fecdf320ea67a72ef
"""
from efficientnet_pytorch import EfficientNet
import numpy as np

a = [1, 2, 1]
b = [1, 2, 1, 1]
print(a + b)
# 补0，这样1月对应日期就是day_in_month[1]
day_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
a = [[1, 2, 3], [2, 3]]
d = {"1": 1, "2": 2}
b = np.array(a, dtype=object)
np.save("test.npy", b)
x = np.load("test.npy", allow_pickle=True)
x = x.tolist()
print(x)
# 计算校验和，输入为身份证id，用string形式
# def get_checksum(id_number:str)->int:
#     checksum = 0
#     for i, number in enumerate(id_number[::-1]):
#         if number == 'x':
#             number = 10
#         number = int(number)
#         weight = 2 ** i
#         checksum += number * weight
#     return checksum % 11
#
# def get_lastdigit(id_number:str)->str:
#     checksum = 0
#     for i, number in enumerate(id_number[:-1:-1]):
#         checksum += int(number) * 2 ** i
#     last_digit = (12 - checksum % 11) % 11
#     if last_digit == 10:
#         last_digit = 'x'
#     return str(last_digit)
#
# # 已知的身份证号部分
# id_number_front = '1301031999'
# id_number_back = '0931'
# # 遍历月份和日期
# for month in range(1, 13):
#     for day in range(1, day_in_month[month]):
#         # 因为身份证号中，1月为'01'，而直接转换int为string，1为'1'，所以小于10的数要补0
#         str_month = '0' + str(month) if month < 10 else str(month)
#         str_day = '0' + str(day) if day < 10 else str(day)
#         id_number = id_number_front + str_month + str_day + id_number_back
#         if get_checksum(id_number) == 1:
#             print('%d月%d日' % (month, day))
#             if get_lastdigit(id_number) == id_number[-1]:
#                 print('%d月%d日OK' % (month, day))
