import copy
import torch
import numpy as np


def para_diff_cal(w_global, w_local):  # 两个OrderedDict做差值
    update = copy.deepcopy(w_global)
    for key, value in w_local.items():
        if key in update:
            update[key] = update[key] - value
        else:
            update[key] = 0 - value
    return update

def float_mulpty_OrderedDict(float_num, weight):
    result = copy.deepcopy(weight)
    for key, value in result.items():
        result[key] = float_num * value
    return result


def OrderedDict_divide_float(float_num, weight):
    if float_num == 0:
        return False
    result = copy.deepcopy(weight)
    for key, value in result.items():
        result[key] = value / float_num
    return result

def normal(delta_ws):  # OrderedDict求二范数
    sum = 0.0
    for key, value in delta_ws.items():
        sum += delta_ws[key].pow(2).sum()
    return float(sum)

def normal_test(delta_ws):  # OrderedDict求二范数
    a = copy.deepcopy(delta_ws)
    sum = []
    for key, value in a.items():
        sum.append(value)
    # print(sum)
    sum_cpu = [i.cpu() for i in sum]
    # print(sum_cpu)
    client_grads = sum_cpu[0] # shape now: (784, 26)

    for i in range(1, len(sum_cpu)):
        client_grads = np.append(client_grads, sum_cpu[i])
    return np.sum(np.square(client_grads))
    

def orderdict_sum(orderdict_lists):  # OrderedDict的列表求和
    result = orderdict_zeros_like(orderdict_lists[0])
    for i in orderdict_lists:
        for key, value in i.items():
            if key in result:
                result[key] += value
            else:
                result[key] =value
    return result 

def orderdict_sum_test(orderdict_lists):  # OrderedDict的列表求和
    result = orderdict_zeros_like(orderdict_lists[0])
    for i in orderdict_lists:
        result = ms_cal(i, result)
    return result 

def orderdict_zeros_like(orderdict):
    ms = copy.deepcopy(orderdict)
    for key, value in orderdict.items():
        ms[key] = torch.zeros_like(value)
    return ms

def delta_kt_sum(weight_lists):  # OrderedDict列表求和
    result = orderdict_zeros_like(weight_lists[0])
    for i in weight_lists:
        for key, value in i.items():
            if key in result:
                result[key] += value
            else:
                result[key] =value
    return result


def ms_cal(a, b):  # OrderedDict的列表求和
    result = copy.deepcopy(a)
    for key, value in b.items():
        if key in result:
            result[key] = result[key] + value
        else:
            result[key] = 0 - value
    return result

def weight_sum(weight_lists):  # OrderedDict的列表求和
    result={}
    for i in weight_lists:
        for key, value in i.items():
            if key in result:
                result[key] += value
            else:
                result[key] =value
    return result

def float_mulpty_OrderedDict(float_num, weight):
    for key, value in weight.items():
        weight[key] = float_num * value
    return weight