from __future__ import print_function
import argparse

import chainer
from chainer import cuda
from chainer import optimizers
import numpy as np
import six

import aggregator
import data
import feeder
import net


parser = argparse.ArgumentParser(description='ADGM')
parser.add_argument('--gpu', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batchsize', default=128, type=int, help='batchsize')
parser.add_argument('--iteration', default=100000, type=int, help='iteration')
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
train_data = feeder.DataFeeder((x_train, y_train))
test_data = feeder.DataFeeder((x_test, y_test))
unlabeled_data = feeder.DataFeeder(x_unlabeled)
N_test = len(test_data)


model = net.ADGM(D, args.a_dim, T, args.z_dim, args.h_dim)
if args.gpu >= 0:
    model.to_gpu(args.gpu)
xp = cuda.cupy if args.gpu >= 0 else np


optimizer = optimizers.Adam()
optimizer.setup(model)


def onehot(y, T):
    ret = xp.zeros((len(y), T), dtype=np.float32)
    ret[:, y] = 1
    return ret


def to_variable(xs, volatile='off'):
    return [chainer.Variable(x, volatile=volatile) for x in xs]


train_loss = aggregator.Aggregator()
train_accuracy = aggregator.Aggregator()
unlabeled_loss = aggregator.Aggregator()
for iteration in six.moves.range(args.iteration):
    if (iteration + 1) % 100 == 0:
        print('iteration\t{}'.format(iteration))

    # Supervised training
    model.train = True
    (x, y), _, _ = train_data.get_minibatch()
    batchsize = len(x)
    y_onehot = onehot(y, T)
    xs = to_variable((x, y, y_onehot))
    optimizer.update(model, *xs)

    train_loss.sum += model.loss * batchsize
    train_loss.n += batchsize
    train_accuracy.sum += float(model.accuracy(xs[0], xs[1]).data) * batchsize
    train_accuracy.n += batchsize

    if (iteration + 1) % 100 == 0:
        print('labeled\tloss\t{}\taccuracy\t{}'.format(train_loss.mean(),
                                                       train_accuracy.mean()))
        train_loss.reset()
        train_accuracy.reset()

    # Unsupervised training
    x, _, _ = unlabeled_data.get_minibatch()
    batchsize = len(x)
    x, = to_variable((x,))
    batchsize = len(x.data)
    optimizer.update(model, x)

    unlabeled_loss.sum += model.loss * batchsize
    unlabeled_loss.n += batchsize

    if (iteration + 1) % 100 == 0:
        print('unlabeled\tloss\t{}'.format(unlabeled_loss.mean()))
        unlabeled_loss.reset()

    # Test
    if (iteration + 1) % 100 == 0:
        model.train = False
        test_loss = aggregator.Aggregator()
        test_accuracy = aggregator.Aggregator()
        for i in six.moves.range(0, N_test, args.batchsize):
            (x, y), _, _ = test_data.get_minibatch()
            y_onehot = onehot(y, T)
            xs = to_variable((x, y, y_onehot), 'on')

            test_loss.sum += float(model(*xs).data) * batchsize
            test_loss.n += batchsize
            accuracy = float(model.accuracy(xs[0], xs[1]).data) * batchsize
            test_accuracy.sum += accuracy
            test_accuracy.n += batchsize
        print('test\tloss\t{}\taccuracy\t{}'.format(test_loss.mean(),
                                                    test_accuracy.mean()))
