import numpy as np
import six
from sklearn.datasets import mldata


def prune_by_stddev(x, threshold=0.1):
    stddev = np.std(x, axis=0)
    return x[:, stddev > threshold]

def load_mnist(N_labeled=100, N_test=10000, pruning=False):
    data = mldata.fetch_mldata('MNIST original')
    x = data['data'].astype(np.float32) / 255
    y = data['target'].astype(np.int32)

    if pruning:
        x = prune_by_stddev(x)
    D = len(x[0])

    T = 10
    N_labeled /= T

    x_split = [np.split(x[y == i], [N_labeled]) for i in six.moves.range(T)]
    x_train = np.concatenate([x_[0] for x_ in x_split])
    x_rest = np.concatenate([x_[1] for x_ in x_split])
    y_split = [np.split(y[y == i], [N_labeled]) for i in six.moves.range(T)]
    y_train = np.concatenate([y_[0] for y_ in y_split])
    y_rest = np.concatenate([y_[1] for y_ in y_split])

    N = 70000
    N_rest = N - N_labeled * T
    perm = np.random.permutation(N_rest)
    x_unlabeled, x_test = np.split(x_rest[perm], [N_rest - N_test])
    _, y_test = np.split(y_rest[perm], [N_rest - N_test])
    return x_train, y_train, x_test, y_test, x_unlabeled, D, T
