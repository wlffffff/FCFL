import numpy as np


def is_all_zero(lst):
    return all(element == 0 for element in lst)

def zero_num(lst):
    num = 0
    for i in lst:
        if i == 0:
            num += 1
    return num

def non_zero_index(lst):
    index = []
    for i in range(len(lst)):
        if lst[i] != 0:
            index.append(i)
    return index

def sample_by_Q_top_add_random(num_clients_per_round, Q_value, ratio):
    if (is_all_zero(Q_value)):
        # print("all zero, random select!")
        result = np.random.choice(range(len(Q_value)), num_clients_per_round, replace=False)
    elif(len(Q_value) - zero_num(Q_value) < num_clients_per_round):
        # print("top + random!")
        sorted_indices = [i[0] for i in sorted(enumerate(Q_value), reverse=True, key=lambda x: x[-1])]
        result = [sorted_indices[i] for i in range(len(Q_value) - zero_num(Q_value))]
        index = non_zero_index(Q_value)
        all = [i for i in range(len(Q_value))]
        select_lst = list(set(all) - set(index))
        result_add = np.random.choice(select_lst, num_clients_per_round - (len(Q_value) - zero_num(Q_value)), replace=False)
        for item in result_add:
            result.append(item)
    else:
        # 对列表进行排序
        sorted_numbers = sorted(Q_value, reverse=True)
        Q_num = int(num_clients_per_round*ratio)
        random_num = num_clients_per_round-Q_num
        # print(sorted_numbers)
        # 获取排序后的索引值
        sorted_indices = [i[0] for i in sorted(enumerate(Q_value), reverse=True, key=lambda x: x[-1])]
        result = [sorted_indices[i] for i in range(Q_num)]

        all = [i for i in range(len(Q_value))]
        select_lst = list(set(all) - set(result))
        result_add = np.random.choice(select_lst, random_num, replace=False)
        for item in result_add:
            result.append(item)
    return result

def sample_by_Q_top(num_clients_per_round, Q_value):
    if (is_all_zero(Q_value)):
        # print("all zero, random select!")
        result = np.random.choice(range(len(Q_value)), num_clients_per_round, replace=False)
    elif(len(Q_value) - zero_num(Q_value) < num_clients_per_round):
        # print("top + random!")
        sorted_indices = [i[0] for i in sorted(enumerate(Q_value), reverse=True, key=lambda x: x[-1])]
        result = [sorted_indices[i] for i in range(len(Q_value) - zero_num(Q_value))]
        index = non_zero_index(Q_value)
        all = [i for i in range(len(Q_value))]
        select_lst = list(set(all) - set(index))
        result_add = np.random.choice(select_lst, num_clients_per_round - (len(Q_value) - zero_num(Q_value)), replace=False)
        for item in result_add:
            result.append(item)
    else:
        # 对列表进行排序
        sorted_numbers = sorted(Q_value, reverse=True)
        # print(sorted_numbers)
        # 获取排序后的索引值
        sorted_indices = [i[0] for i in sorted(enumerate(Q_value), reverse=True, key=lambda x: x[-1])]
        result = [sorted_indices[i] for i in range(num_clients_per_round)]
    return result


def sample_by_Q_prob(num_clients_per_round, Q_value):
    if (is_all_zero(Q_value)):
        result = np.random.choice(range(len(Q_value)), num_clients_per_round, replace=False)
    else:
        sum_Q_value = np.sum(Q_value)
        probability = [Q/sum_Q_value for Q in Q_value]
        result = np.random.choice(range(len(Q_value)), num_clients_per_round, replace=False, p = probability)
    return result


def acc_global_estimate(acc_list, weight_list):
    result = 0.0
    num = len(acc_list)
    for i in range(num):
        result = result + acc_list[i] * weight_list[i]
    return result