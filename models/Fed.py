#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
from utils.weight_cal import weight_sum, para_diff_cal, delta_kt_sum, float_mulpty_OrderedDict
import collections


def FedAvg(w):   # main中将w_locals赋给w，即worker计算出的权值
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():  # 对于每个参与的设备：
        for i in range(1, len(w)):  # 对本地更新进行聚合
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def FedWeightAvg(w, size):
    totalSize = sum(size)
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w[0][k]*size[0]
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * size[i]
        # print(w_avg[k])
        w_avg[k] = torch.div(w_avg[k], totalSize)
    return w_avg

def weight_agg(model_list, weight_list):
    new_model_key = model_list[0].keys()
    new_model_value = [float_mulpty_OrderedDict(j, i) for i, j in zip(model_list, weight_list)]
    # new_model = dict(zip(new_model_key, new_model_value))
    new_model_value = weight_sum(new_model_value)
    # print(new_model_value)
    # print(collections.OrderedDict(new_model_value))
    return collections.OrderedDict(new_model_value)
