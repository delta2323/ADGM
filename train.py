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
parser.add_argument('--a-dim', default=100, type=int)
parser.add_argument('--z-dim', default=100, type=int)
parser.add_argument('--h-dim', default=500, type=int)
parser.add_argument('--beta', default=1, type=float)
args = parser.parse_args()


chainer.set_debug(True)


np.random.seed(args.seed)
if args.gpu >= 0:
    cuda.cupy.random.seed(args.seed)


D = 784
T = 10


x_labeled, y_labeled, x_test, y_test, x_unlabeled = data.load_mnist()
labeled_data = feeder.DataFeeder((x_labeled, y_labeled))
test_data = feeder.DataFeeder((x_test, y_test))
unlabeled_data = feeder.DataFeeder(x_unlabeled)
N_labeled = len(labeled_data)
N_unlabeled = len(unlabeled_data)

gamma = args.beta * float(N_labeled + N_unlabeled) / N_labeled
model = net.ADGM(D, args.a_dim, T, args.z_dim, args.h_dim, gamma)
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


labeled_loss = aggregator.Aggregator()
labeled_accuracy = aggregator.Aggregator()
unlabeled_loss = aggregator.Aggregator()
for iteration in six.moves.range(args.iteration):
    if (iteration + 1) % 10 == 0:
        print('iteration\t{}'.format(iteration))

    # Supervised training
    model.train = True
    (x, y), _ = labeled_data.get_minibatch()
    batchsize = len(x)
    y_onehot = onehot(y, T)
    xs = to_variable((x, y, y_onehot))
    optimizer.update(model, *xs)

    labeled_loss.sum += model.loss * batchsize
    labeled_loss.n += batchsize
    labeled_accuracy.sum += float(model.accuracy(xs[0], xs[1]).data) * batchsize
    labeled_accuracy.n += batchsize

    if (iteration + 1) % 10 == 0:
        print('labeled\tloss\t{}\taccuracy\t{}'.format(labeled_loss.mean(),
                                                       labeled_accuracy.mean()))
        labeled_loss.reset()
        labeled_accuracy.reset()

    # Unsupervised training
    x, _ = unlabeled_data.get_minibatch()
    batchsize = len(x)
    x, = to_variable((x,))
    batchsize = len(x.data)
    optimizer.update(model, x)

    unlabeled_loss.sum += model.loss * batchsize
    unlabeled_loss.n += batchsize

    if (iteration + 1) % 10 == 0:
        print('unlabeled\tloss\t{}'.format(unlabeled_loss.mean()))
        unlabeled_loss.reset()

    # Test
    if (iteration + 1) % 10 == 0:
        model.train = False
        test_loss = aggregator.Aggregator()
        test_accuracy = aggregator.Aggregator()
        test_data.reset()
        while not test_data.exhaust:
            (x, y), _ = test_data.get_minibatch()
            y_onehot = onehot(y, T)
            xs = to_variable((x, y, y_onehot), 'on')

            test_loss.sum += float(model(*xs).data) * batchsize
            test_loss.n += batchsize
            accuracy = float(model.accuracy(xs[0], xs[1]).data) * batchsize
            test_accuracy.sum += accuracy
            test_accuracy.n += batchsize
        print('test\tloss\t{}\taccuracy\t{}'.format(test_loss.mean(),
                                                    test_accuracy.mean()))
