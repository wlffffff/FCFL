import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from datetime import datetime
from time import strftime
import random
import os
import time
import pickle

from client_split_sample.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from utils.options import args_parser
from client_split_sample.Dirichlet_split_datasets import split_noniid
from client_split_sample.sampling_by_proportion import sample_by_proportion
from models.Update import LocalUpdate
from utils.weight_cal import para_diff_cal, float_mulpty_OrderedDict, normal_test
from models.Nets import MLP, CNNMnist, CNNCifar, CharLSTM
from models.qFed import weight_agg
from utils.dataset import ShakeSpeare
from models.test import test_img
from utils.utils import exp_details, worst_fraction, best_fraction

from utils.FCFL_options import sample_by_Q_top, is_all_zero, acc_global_estimate


if __name__ == '__main__':
     # parse args  # python自带的命令行参数解析包，读取命令行参数
    args = args_parser()   
    args.method = "fcfl"
    exp_details(args)  # 打印超参数
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu') # 使用cpu还是gpu 赋值args.device
    print(args.device)
    print(torch.cuda.is_available())

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)  #训练集
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)  #测试集
        # sample users
        if args.iid:
            np.random.seed(args.seed)
            dict_users = mnist_iid(dataset_train, args.num_users)  # 为用户分配iid数据
        else:
            if args.dirichlet != 0:
                print("dirichlet")
                np.random.seed(args.seed)
                labels_train = np.array(dataset_train.targets)
                dict_users = split_noniid(labels_train, args.dirichlet, args.num_users)
            else:
                print("shard")
                np.random.seed(args.seed)
                dict_users = mnist_noniid(dataset_train, args.num_users)  # 否则为用户分配non-iid数据
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar)  #训练集
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar)  #测试集
        if args.iid:
            np.random.seed(args.seed)
            dict_users = cifar_iid(dataset_train, args.num_users)  # 为用户分配iid数据
        else:
            if args.dirichlet != 0:
                print("dirichlet")
                np.random.seed(args.seed)
                labels_train = np.array(dataset_train.targets)
                dict_users = split_noniid(labels_train, args.dirichlet, args.num_users)
            else:
                print("shard")
                np.random.seed(args.seed)
                dict_users = cifar_noniid(dataset_train, args.num_users)
    elif args.dataset == 'shakespeare':
        dataset_train = ShakeSpeare(train=True)
        dataset_test = ShakeSpeare(train=False)
        dict_users = dataset_train.get_client_dic()
        args.num_users = len(dict_users)
        if args.iid:
            exit('Error: ShakeSpeare dataset is naturally non-iid')
        else:
            print("Warning: The ShakeSpeare dataset is naturally non-iid, you do not need to specify iid or non-iid")
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape  # 图像的size

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.dataset == 'shakespeare' and args.model == 'lstm':
        net_glob = CharLSTM().to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)

    net_glob.train()
    w_glob = net_glob.state_dict()   
    # training
    loss_train = []
    
    for iter in range(args.epochs):
        loss_locals = []  # 对于每一个epoch，初始化worker的损失
        w_locals = []  # 存储客户端本地权重

        local_weights, local_losses, local_accs_testloader, local_accs_trainloader = [], [], [], []

        for user_i in range(args.num_users):    # 全局模型在每个用户testloader上的性能
            local_model = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[user_i])
            acc_test, _ = local_model.inference_testloader(net_glob)
            local_accs_testloader.append(acc_test)

        net_glob.train()
        if iter == 0:
            Q = np.zeros((args.num_users, args.epochs), float)   # 初始化不公平累计队列Q
            for i in range(args.num_users):
                Q[i][0] = 0.0
            # print(Q)
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)

            local_data_volume = [len(dict_users[cid]) for cid in range(len(idxs_users))]
            total_data_volume = sum(local_data_volume)
            aggregate_p = [l_d_v / total_data_volume for l_d_v in local_data_volume]
        else:
            # 计算uf
            uf = [0.0] * args.num_users
            for i in range(args.num_users):
                uf[i] = Acc_global - local_accs_testloader[i] if Acc_global > local_accs_testloader[i] else 0
            # print(uf)
            # print(min(uf))
            # 更新Q
            Q_value = []
            aggregate_p_all_clients = [0.0] * args.num_users
            # print(weights)
            for i, j in zip(idxs_users, range(m)):
                aggregate_p_all_clients[i] = aggregate_p[j]
            # print(aggregate_p_all_clients)
            # print(weights)

            for i in range(args.num_users):
                Q[i][iter] = max(Q[i][iter-1] + args.alpha * uf[i] - aggregate_p_all_clients[i], 0)
                Q_value.append(Q[i][iter])
            # print(Q)
            # 选择客户端
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = sample_by_Q_top(m, Q_value)  # 根据Q选取top-m个客户端
            sum_Q = sum(Q[i][iter] for i in idxs_users)
            # print(weights)
            aggregate_p = []
            # print(weights)
            if sum_Q == 0 and is_all_zero(Q[:, iter]):
                local_data_volume = [len(dict_users[cid]) for cid in range(len(idxs_users))]
                total_data_volume = sum(local_data_volume)
                aggregate_p = [l_d_v / total_data_volume for l_d_v in local_data_volume]
            else:
                for i in idxs_users:
                    weight_clients = Q[i][iter]/sum_Q
                    aggregate_p.append(weight_clients)

        
        for idx in idxs_users:  # 对于选取的m个worker
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx]) # 对每个worker进行本地更新
            w, loss, acc_trainloader = local.fcfl_train(net=copy.deepcopy(net_glob).to(args.device)) # 本地训练的weight和loss  ##第5行完成

            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            local_accs_trainloader.append(acc_trainloader)

        w_glob = weight_agg(w_locals, aggregate_p)
        net_glob.load_state_dict(w_glob)
        Acc_global = acc_global_estimate(local_accs_trainloader, aggregate_p)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    # plt.savefig('./save/fcfl_{}_{}_{}_C{}_iid{}_{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid, datetime.now().strftime("%H_%M_%S")))

    # testing  测试集上进行测试
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)

    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

    all_user_acc = []  # 测试全局模型在所有客户端上的性能，计算方差
    
    with open('./save/result/Result_FCFL.txt', 'a') as f:
        f.truncate(0)
    for user in range(args.num_users):   
        local_model = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[user])
        acc_user, _ = local_model.inference_testloader(net_glob)
        all_user_acc.append(acc_user*100)
        with open('./save/result/Result_FCFL.txt', 'a') as f:
            f.write(str(user) + ': Local testloader accuracy: ' + str(float(acc_user)) + '\n')

    """
    5次求平均
    """
    with open('./save/result/Result_FCFL_5_avg.txt', 'a') as f:
        print("yes")
        f.write(str(float(acc_test))+'\n')
        f.write(str(worst_fraction(all_user_acc, 0.1))+'\n')
        f.write(str(best_fraction(all_user_acc, 0.1))+'\n')
        f.write(str(np.var(all_user_acc))+'\n')
        f.write('\n')
    """
    """
    
    print(worst_fraction(all_user_acc, 0.1))
    print(best_fraction(all_user_acc, 0.1))
    print(np.var(all_user_acc))
