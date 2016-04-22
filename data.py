import numpy as np
import six
from sklearn.datasets import mldata


def load_mnist(N_unlabeled=100, N_test=10000):
    data = mldata.fetch_mldata('MNIST original')
    x = data['data'].astype(np.float32) / 255
    y = data['target'].astype(np.int32)

    T = 10
    N_unlabeled /= T

    x_split = [np.split(x[y == i], [N_unlabeled]) for i in six.moves.range(T)]
    x_train = np.concatenate([x[0] for x in x_split])
    x_rest = np.concatenate([x[1] for x in x_split])
    y_split = [np.split(y[y == i], [N_unlabeled]) for i in six.moves.range(T)]
    y_train = np.concatenate([y[0] for y in y_split])
    y_rest = np.concatenate([y[1] for y in y_split])

    N = 70000
    N_rest = N - unlabeled * T
    perm = np.random.permutation(N_rest)
    x_unlabeled, x_test = np.split(x_rest[perm], [N_rest - N_test])
    _, y_test = np.split(y_rest[perm], [N_rest - N_test])
    return x_train, y_train, x_test, y_test, x_unlabeled
