#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()  # 创建一个解析器（Argument Parser()对象）
    # federated arguments     # 添加参数 调用add_argument()方法添加参数
    parser.add_argument('--epochs', type=int, default=20, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=64, help="local batch size: B")
    parser.add_argument('--bs_test', type=int, default=10, help="test batch size")
    parser.add_argument('--bs_train', type=int, default=64, help="train batch size")
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    parser.add_argument('--dirichlet', type=float, default=0.0, help="dirichlet ratio to split dataset")

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', type=int, default=0,help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')
    parser.add_argument('--sample_by_proportion', default=False, action='store_true', help='sample by number of local datasets')

    parser.add_argument('--q', type=float,default=1.0, help='parameter of q-fedavg "q"')
    parser.add_argument('--beta', type=float, default=0.5, help='parameter of fedfa "beta"')
    parser.add_argument('--gamma', type=float, default=0.9, help='parameter of fedfa "gamma"')

    parser.add_argument('--alpha', type=float, default=0.5, help='parameter of fcfl "alpha"')
    parser.add_argument('--ratio', type=float, default=0.7, help='parameter of fcfl "ratio"')

    args = parser.parse_args()  # 解析参数 使用parse_args()解析添加的参数
    return args
