#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
import numpy as np
from utils.weight_cal import weight_sum, para_diff_cal, delta_kt_sum, float_mulpty_OrderedDict, OrderedDict_divide_float, orderdict_sum
import collections

def qfedavg(global_model, delta_ks, hks):   # main中将w_locals赋给w，即worker计算出的权值
    # keys = global_model.keys()
    
    # sum_delta_kt = delta_kt_sum(delta_ks)
    # hks_cpu = [i.cpu() for i in hks]
    new_model = copy.deepcopy(global_model)
    update = []
    sum_h_kt = np.sum(np.asarray(hks))
    for i in delta_ks:
        update.append(OrderedDict_divide_float(sum_h_kt, i))
    values = orderdict_sum(update)
    # new_model = dict(zip(keys, global_model))
    return para_diff_cal(new_model, values)


def q_aggregate(server_model, deltas, hs):
    num_clients = len(deltas)
    de = np.sum(np.asarray(hs))
    # Scale client deltas by multiplying (1/denominator)
    scaled_deltas = []
    for client_delta in deltas:
        scaled_deltas.append([(layer * 1.0 / de) for layer in client_delta])

    # Sum scaled client deltas
    sum_delta = deltas[0]
    for i in range(len(scaled_deltas)):
        if(i > 0):
            sum_delta = [(sd+d) for sd, d in zip(sum_delta, scaled_deltas[i])]
 
    # Update server model
    model = copy.deepcopy(server_model)

    with torch.no_grad():
        model.linear.weight -= sum_delta[0]
        model.linear.bias -=  sum_delta[1]
    return model


# def FedAvg(w):   # main中将w_locals赋给w，即worker计算出的权值
#     w_avg = copy.deepcopy(w[0])
#     for k in w_avg.keys():  # 对于每个参与的设备：
#         for i in range(1, len(w)):  # 对本地更新进行聚合
#             w_avg[k] += w[i][k]
#         w_avg[k] = torch.div(w_avg[k], len(w))
#     return w_avg


def FedAvg(w):   # main中将w_locals赋给w，即worker计算出的权值
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():  # 对于每个参与的设备：
        for i in range(1, len(w)):  # 对本地更新进行聚合
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def weight_agg(model_list, weight_list):
    new_model_key = model_list[0].keys()
    new_model_value = [float_mulpty_OrderedDict(j, i) for i, j in zip(model_list, weight_list)]
    # new_model = dict(zip(new_model_key, new_model_value))
    new_model_value = weight_sum(new_model_value)
    # print(new_model_value)
    # print(collections.OrderedDict(new_model_value))
    return collections.OrderedDict(new_model_value)

def margin_glob_model(model_list, weight_list):
    margin_glob_model = []
    length = len(model_list)
    for i in range(length):
        del_model_list = copy.deepcopy(model_list)
        del del_model_list[i]
        del_weight_list = copy.deepcopy(weight_list)
        del del_weight_list[i]
        new_model_value = [float_mulpty_OrderedDict(j, i) for i, j in zip(del_model_list, del_weight_list)]
        new_model_value = weight_sum(new_model_value)
        margin_glob_model.append(collections.OrderedDict(new_model_value))
    return margin_glob_model



def q_FedAvg(global_model, delta_ks, hks):   # main中将w_locals赋给w，即worker计算出的权值
    keys = global_model.keys()
    
    sum_delta_kt = delta_kt_sum(delta_ks)
    sum_h_kt = np.sum(np.asarray(hks))
    values = [tensor_value/sum_h_kt for tensor_value in sum_delta_kt]
    global_model = [(x - y) for x, y in zip(global_model.values(), values)]

    new_model = dict(zip(keys, global_model))
    return new_model
