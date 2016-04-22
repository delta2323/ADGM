from __future__ import print_function
import argparse

import chainer
from chainer import cuda
from chainer import optimizers
import numpy as np
import six

import data
import net


parser = argparse.ArgumentParser(description='ADGM')
parser.add_argument('--gpu', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batchsize', default=128, type=int, help='batchsize')
parser.add_argument('--epoch', default=10, type=int, help='epoch')
parser.add_argument('--a-dim', default=10, type=int)
parser.add_argument('--z-dim', default=10, type=int)
args = parser.parse_args()

np.random.seed(args.seed)
if args.gpu >= 0:
    cuda.cupy.random.seed(args.seed)

D = 784
T = 10

model = net.ADGM(D, args.a_dim, T, args.z_dim)
if args.gpu >= 0:
    model.to_gpu(args.gpu)
xp = cuda.cupy if args.gpu >= 0 else np


x_train, y_train, x_test, y_test, x_unlabeled, _ = data.load_mnist()
N_train = len(x_train)
N_test = len(x_test)
N_unlabeled = len(x_unlabeled)


def next_minibatch(batchsize, *xs_data):
    xs = [xp.asarray(x_data[i: i + batchsize]) for x_data in xs_data]
    return [chainer.Variable(x) for x in xs]


def onehot(y, T):
    ret = xp.zeros((len(y), T), dtype=np.float32)
    ret[:, y] = 1
    return ret


for epoch in six.moves.range(args.epoch):
    model.train = True
    loss = 0.0
    for i in six.moves.range(args.batchsize, N_train):
        x, y = next_minibatch(args.batchsize, x_train, y_train)
        y = onehot(y, T)
        optimizer.update(model, x, y)
        loss += model.loss * len(x.data)
    loss /= N_train
    print('epoch\t{}\tsupervised loss\t{}'.format(epoch, loss))


    loss = 0.0
    for i in six.moves.range(args.batchsize, N_unsupervised):
        x, = next_minibatch(args.batchsize, x_unlabeled)
        optimizer.update(model, x)
        loss += model.loss * len(x.data)
    loss /= N_unsupervised
    print('epoch\t{}\tunsupervised loss\t{}'.format(epoch, loss))


    model.train = False
    loss = 0.0
    for i in six.moves.range(args.batchsize, N_test):
        x, y = next_minibatch(args.batchsize, x_test, y_test)
        y = onehot(y, T)
        loss += model(x, y) * len(x.data)
    loss /= N_test
    print('epoch\t{}\ttest loss\t{}'.format(epoch, loss))

