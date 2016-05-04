import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import six

import loss as l


def entropy(p):
    return -F.sum(F.log(p) * p)


def split(x):
    return F.split_axis(x, 2, 1)


class ADGM(chainer.Chain):

    def __init__(self, x_dim, a_dim, y_dim, z_dim, h_dim,
                 gamma, sampling_num=1):
        q_a_given_x = MLP((x_dim, h_dim, h_dim, a_dim * 2))
        q_z_given_a_y_x = MultiMLP((a_dim, y_dim, x_dim),
                                   (h_dim, h_dim, z_dim * 2))
        q_y_given_a_x = MultiMLP((a_dim, x_dim), (h_dim, h_dim, y_dim))
        p_a_given_z_y_x = MultiMLP((z_dim, y_dim, x_dim),
                                   (h_dim, h_dim, a_dim * 2))
        p_x_given_z_y = MultiMLP((z_dim, y_dim),
                                 (h_dim, h_dim, x_dim * 2))
        super(ADGM, self).__init__(
            q_a_given_x=q_a_given_x,
            q_z_given_a_y_x=q_z_given_a_y_x,
            q_y_given_a_x=q_y_given_a_x,
            p_a_given_z_y_x=p_a_given_z_y_x,
            p_x_given_z_y=p_x_given_z_y)
        self.verbose = False
        # gamma corresponds to beta * (Nl + Nu) / Nl in the original paper.
        self.gamma = gamma
        self.sampling_num = sampling_num
        self.y_dim = y_dim

    @property
    def train(self):
        return self.q_a_given_x.train

    @train.setter
    def train(self, val):
        for c in self._children:
            self.__dict__[c].train = val

    def loss_z_dep(self, x, y, a):
        def to_onehot(y, T):
            ret = np.zeros((len(y), T), dtype=np.float32)
            ret[:, y.get()] = 1.0
            return chainer.Variable(self.xp.asarray(ret), volatile='auto')

        y = to_onehot(y.data, self.y_dim)
        z_mean, z_ln_var = split(self.q_z_given_a_y_x(a, y, x))
        z = F.gaussian(z_mean, z_ln_var)
        a_mean, a_ln_var = split(self.p_a_given_z_y_x(z, y, x))
        x_mean, _ = split(self.p_x_given_z_y(z, y))
        zero = chainer.Variable(self.xp.zeros_like(z.data), volatile='auto')

        nll_p_z = F.sum(l.gaussian_nll(z, zero, zero), axis=1)
        nll_p_x_given_z_y = F.sum(l.bernoulli_nll(x, x_mean), axis=1)
        nll_p_a_given_z_y_x = F.sum(
            l.gaussian_nll(a, a_mean, a_ln_var), axis=1)
        nll_q_z_given_a_y_x = F.sum(
            l.gaussian_nll(z, z_mean, z_ln_var), axis=1)

        return (nll_p_z + nll_p_x_given_z_y +
                nll_p_a_given_z_y_x - nll_q_z_given_a_y_x)

    def predict(self, x, softmax=False):
        a_mean, _ = split(self.q_a_given_x(x))
        y_pred = self.q_y_given_a_x(a_mean, x)
        if softmax:
            y_pred = F.softmax(y_pred)
        return y_pred

    def classification_loss(self, x, y):
        y_pred = self.predict(x)
        return F.softmax_cross_entropy(y_pred, y)

    def accuracy(self, x, y):
        y_pred = self.predict(x)
        return F.accuracy(y_pred, y)

    def loss_one(self, x, y=None):
        a_mean, a_ln_var = split(self.q_a_given_x(x))
        a = F.gaussian(a_mean, a_ln_var)

        # nll_q_a_given_x
        loss = -F.sum(l.gaussian_nll(a, a_mean, a_ln_var))
        loss += np.log(self.y_dim)  # nll_p_given_y
        if y is None:
            losses_z_dep = []
            for i in six.moves.range(self.y_dim):
                y_ = chainer.Variable(
                    self.xp.full((len(x.data), self.y_dim), i, dtype=np.int32),
                    volatile='auto')
                loss_z_dep = self.loss_z_dep(x, y_, a)
                loss_z_dep = F.reshape(loss_z_dep, (-1, 1))
                losses_z_dep.append(loss_z_dep)
            q_y_given_a_x = F.softmax(self.q_y_given_a_x(a, x))
            loss -= entropy(q_y_given_a_x)  # nll_q_y_given_a_x
            # nll_p_z + nll_p_x_given_z_y + nll_p_a_given_x_y_z - nll_q_z_given_a_y_x
            loss += F.sum(F.concat(losses_z_dep) * q_y_given_a_x)
        else:
            loss += F.sum(self.loss_z_dep(x, y, a))
            loss += self.gamma * self.classification_loss(x, y)
        return loss

    def __call__(self, x, y=None):
        loss = 0.0
        for _ in six.moves.range(self.sampling_num):
            loss += self.loss_one(x, y)
        loss /= self.sampling_num
        self.loss = float(loss.data)
        return loss


class MLP(chainer.Chain):

    def __init__(self, units):
        self.layer_num = len(units) - 1
        linear = chainer.ChainList(*[L.Linear(units[i], units[i + 1])
                                     for i in six.moves.range(self.layer_num)])
        bn = chainer.ChainList(*[L.BatchNormalization(units[i + 1])
                                 for i in six.moves.range(self.layer_num - 1)])
        super(MLP, self).__init__(linear=linear, bn=bn)
        self.train = True

    def __call__(self, x):
        for i in six.moves.range(self.layer_num - 1):
            x = self.linear[i](x)
            x = self.bn[i](x, test=not self.train)
            x = F.relu(x)
        return self.linear[self.layer_num - 1](x)


class MultiMLP(MLP):

    def __init__(self, input_dims, units):
        input_dim = np.sum(input_dims)
        super(MultiMLP, self).__init__((input_dim,) + units)

    def __call__(self, *xs):
        x = F.concat(xs)
        return super(MultiMLP, self).__call__(x)
