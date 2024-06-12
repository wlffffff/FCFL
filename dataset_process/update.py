import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
from torch.autograd import Variable
import torch.nn.functional as F
    

def split(x_train, y_train, idxs):
        image = []
        label = []
        for i in idxs:
            # print(i)
            image.append(x_train[i])
            label.append(y_train[i])
        image = torch.tensor(image)
        image = image.reshape(len(image),x_train.shape[1],x_train.shape[2],x_train.shape[3])
        label = torch.tensor(label)
        # print(image.shape)   # torch.Size([600, 1, 28, 28])
        # print(label.shape)   # torch.Size([600])
        return image, label


def local_train(net_glob, x_train, y_train, args):  
        net_glob.train()
        optimizer = torch.optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum)
        loss_func = nn.CrossEntropyLoss()
        epoch_loss = []

        for iter in range(args.local_ep):
            batch_loss = []
            num_samples = len(x_train)
            num_batchs = int(num_samples/args.bs_train)

            for k in range(num_batchs):
                start,end = k*args.bs_train, (k+1)*args.bs_train
                data,target = Variable(x_train[start:end],requires_grad=True).to(args.device), Variable(y_train[start:end]).to(args.device)
                # print(data.shape)
                # print(target)
                # images, labels = data.to(args.device), target.to(args.device)
                net_glob.zero_grad()
                log_probs = net_glob(data)
                loss = loss_func(log_probs, target.long())
                # loss = F.cross_entropy(log_probs, target.long())
                loss.backward()
                optimizer.step()
                if args.verbose and k % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(iter, k * len(data), len(x_train),100. * k / len(x_train), loss.item()))
                batch_loss.append(loss.detach().item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

            # Compute loss on the whole training data
            comp_loss = []
            for k in range(num_batchs):
                start,end = k*args.bs_train, (k+1)*args.bs_train
                data,target = Variable(x_train[start:end],requires_grad=True).to(args.device), Variable(y_train[start:end]).to(args.device)
                # print(data.shape)
                # print(target)
                # images, labels = data.to(args.device), target.to(args.device)
                net_glob.zero_grad()
                log_probs = net_glob(data)
                loss = loss_func(log_probs, target.long())
                comp_loss.append(loss.detach().item())
            comp_loss = sum(comp_loss)/len(comp_loss)
        return net_glob.state_dict(), sum(epoch_loss) / len(epoch_loss), comp_loss


