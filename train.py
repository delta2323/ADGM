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
parser.add_argument('--h-dim', default=50, type=int)
args = parser.parse_args()


np.random.seed(args.seed)
if args.gpu >= 0:
    cuda.cupy.random.seed(args.seed)


D = 784
T = 10


x_train, y_train, x_test, y_test, x_unlabeled = data.load_mnist()
N_train = len(x_train)
N_test = len(x_test)
N_unlabeled = len(x_unlabeled)


model = net.ADGM(D, args.a_dim, T, args.z_dim, args.h_dim)
if args.gpu >= 0:
    model.to_gpu(args.gpu)
xp = cuda.cupy if args.gpu >= 0 else np


optimizer = optimizers.Adam()
optimizer.setup(model)


def next_minibatch(batchsize, *xs_data):
    return [xp.asarray(x_data[i: i + batchsize]) for x_data in xs_data]


def onehot(y, T):
    ret = xp.zeros((len(y), T), dtype=np.float32)
    ret[:, y] = 1
    return ret


def to_variable(*xs):
    return [chainer.Variable(x) for x in xs]


for epoch in six.moves.range(args.epoch):
    print('epoch\t{}'.format(epoch))
    model.train = True
    loss, accuracy = 0.0, 0.0
    for i in six.moves.range(0, N_train, args.batchsize):
        x, y = next_minibatch(args.batchsize, x_train, y_train)
        y_onehot = onehot(y, T)
        xs = to_variable(x, y, y_onehot)
        batchsize = len(xs[0].data)

        optimizer.update(model, *xs)

        loss += model.loss * batchsize
        accuracy += float(model.accuracy(xs[0], xs[1]).data) * batchsize
    loss /= N_train
    accuracy /= N_train
    print('labeled\tloss\t{}\taccuracy\t{}'.format(loss, accuracy))

    loss = 0.0
    for i in six.moves.range(0, N_unlabeled, args.batchsize):
        xs = next_minibatch(args.batchsize, x_unlabeled)
        xs = to_variable(*xs)
        batchsize = len(xs[0].data)

        optimizer.update(model, *xs)

        loss += model.loss * batchsize
    loss /= N_unlabeled
    print('unlabeled\tloss\t{}'.format(loss))

    model.train = False
    loss, accuracy = 0.0, 0.0
    for i in six.moves.range(0, N_test, args.batchsize):
        x, y = next_minibatch(args.batchsize, x_test, y_test)
        y_onehot = onehot(y, T)
        xs = to_variable(x, y, y_onehot)
        batchsize = len(xs[0].data)

        loss += float(model(*xs).data) * batchsize
        accuracy += float(model.accuracy(xs[0], xs[1]).data) * batchsize
    loss /= N_test
    accuracy /= N_test
    print('test\tloss\t{}\taccuracy\t{}'.format(loss, accuracy))
