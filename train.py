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
parser.add_argument('--alpha', default=3e-4, type=float)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--binarize', action='store_true')
parser.add_argument('--pruning', action='store_true')
args = parser.parse_args()

if args.debug:
    chainer.set_debug(True)


np.random.seed(args.seed)
if args.gpu >= 0:
    cuda.cupy.random.seed(args.seed)


(x_labeled, y_labeled, x_test, y_test,
 x_unlabeled, D, T) = data.load_mnist(pruning=args.pruning)
labeled_data = feeder.DataFeeder((x_labeled, y_labeled))
test_data = feeder.DataFeeder((x_test, y_test))
unlabeled_data = feeder.DataFeeder(x_unlabeled)
N_labeled = len(labeled_data)
N_unlabeled = len(unlabeled_data)

gamma = args.beta * float(N_labeled + N_unlabeled) / N_labeled
model = net.ADGM(D, args.a_dim, T, args.z_dim, args.h_dim, gamma)
model.verbose = args.verbose
if args.gpu >= 0:
    model.to_gpu(args.gpu)
xp = cuda.cupy if args.gpu >= 0 else np


optimizer = optimizers.Adam(alpha=args.alpha)
optimizer.setup(model)


def to_variable(xs, volatile='off'):
    return [chainer.Variable(xp.asarray(x), volatile=volatile) for x in xs]


def binarize(x):
    rnd = np.random.uniform(size=x.shape)
    return np.where(rnd < x, 1.0, 0.0).astype(np.float32)


labeled_loss = aggregator.Aggregator()
labeled_accuracy = aggregator.Aggregator()
unlabeled_loss = aggregator.Aggregator()
for iteration in six.moves.range(args.iteration):
    # Supervised training
    model.train = True
    (x, y), _ = labeled_data.get_minibatch()
    if args.binarize:
        x = binarize(x)
    batchsize = len(x)
    xs = to_variable((x, y))
    optimizer.update(model, *xs)

    labeled_loss.sum += model.loss
    labeled_loss.n += batchsize
    labeled_accuracy.sum += float(
        model.accuracy(xs[0], xs[1]).data) * batchsize
    labeled_accuracy.n += batchsize

    if (iteration + 1) % 5 == 0:
        print('iteration\t{0}\tlabel\tloss\t{1:3f}\taccuracy\t{2:3f}'.format(
            iteration, labeled_loss.mean(), labeled_accuracy.mean()))
        labeled_loss.reset()
        labeled_accuracy.reset()

    # Unsupervised training
    model.train = True
    x, _ = unlabeled_data.get_minibatch()
    if args.binarize:
        x = binarize(x)
    batchsize = len(x)
    x, = to_variable((x,))
    optimizer.update(model, x)

    unlabeled_loss.sum += model.loss
    unlabeled_loss.n += batchsize

    if (iteration + 1) % 5 == 0:
        print('iteration\t{0}\tunlabel\tloss\t{1:3f}'.format(
            iteration, unlabeled_loss.mean()))
        unlabeled_loss.reset()

    # Test
    if (iteration + 1) % 5 == 0:
        model.train = False
        test_loss = aggregator.Aggregator()
        test_accuracy = aggregator.Aggregator()
        test_data.reset()
        while not test_data.exhaust:
            (x, y), _ = test_data.get_minibatch()
            if args.binarize:
                x = binarize(x)
            batchsize = len(x)
            xs = to_variable((x, y), 'on')

            test_loss.sum += float(model(*xs).data)
            test_loss.n += batchsize
            test_accuracy.sum += float(
                model.accuracy(xs[0], xs[1]).data) * batchsize
            test_accuracy.n += batchsize
        print('iteration\t{0}\ttest\tloss\t{1:3f}\taccuracy\t{2:3f}'.format(
            iteration, test_loss.mean(), test_accuracy.mean()))
