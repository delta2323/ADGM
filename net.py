import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import six


def entropy(p):
    return -F.sum(F.log(p) * p)


def split(x):
    return F.split_axis(x, 2, 1)


class ADGM(chainer.Chain):

    def __init__(self, x_dim, a_dim, y_dim, z_dim, h_dim):
        q_a_given_x = MLP((x_dim, h_dim, a_dim * 2))
        q_z_given_a_y_x = MultiMLP((a_dim, y_dim, x_dim), (h_dim, z_dim * 2))
        q_y_given_a_x = MultiMLP((a_dim, x_dim), (h_dim, y_dim))
        p_a_given_z_y_x = MultiMLP((z_dim, y_dim, x_dim), (h_dim, a_dim * 2))
        p_x_given_z_y = MultiMLP((z_dim, y_dim), (h_dim, x_dim * 2))
        super(ADGM, self).__init__(
            q_a_given_x=q_a_given_x,
            q_z_given_a_y_x=q_z_given_a_y_x,
            q_y_given_a_x=q_y_given_a_x,
            p_a_given_z_y_x=p_a_given_z_y_x,
            p_x_given_z_y=p_x_given_z_y)

    @property
    def train(self):
        return self.q_a_given_x.train

    @train.setter
    def train(self, val):
        for c in self._children:
            self.__dict__[c].train = val

    def ELBO(self, x, y=None):
        a_enc_mean, a_enc_ln_var = split(self.q_a_given_x(x))
        a = F.gaussian(a_enc_mean, a_enc_ln_var)
        if y is None:
            y = F.softmax(self.q_y_given_a_x(a, x))
            nll_q_y_given_a_x = entropy(y)
        else:
            nll_q_y_given_a_x = 0.0

        z_enc_mean, z_enc_ln_var = split(self.q_z_given_a_y_x(a, y, x))
        z = F.gaussian(z_enc_mean, z_enc_ln_var)

        a_dec_mean, a_dec_ln_var = split(self.p_a_given_z_y_x(z, y, x))
        x_dec_mean, x_dec_ln_var = split(self.p_x_given_z_y(z, y))

        zero = chainer.Variable(self.xp.zeros_like(z.data), volatile='auto')
        nll_p_z = F.gaussian_nll(z, zero, zero)
        nll_p_x_given_z_y = F.gaussian_nll(x, x_dec_mean, x_dec_ln_var)
        nll_p_a_given_z_y_x = F.gaussian_nll(a, a_dec_mean, a_dec_ln_var)
        nll_q_a_given_x = F.gaussian_nll(a, a_enc_mean, a_enc_ln_var)
        nll_q_z_given_a_y_x = F.gaussian_nll(z, z_enc_mean, z_enc_ln_var)

        return (nll_p_z + nll_p_x_given_z_y + nll_p_a_given_z_y_x +
                nll_q_y_given_a_x + nll_q_a_given_x + nll_q_z_given_a_y_x)

    def predict(self, x, y, softmax=False):
        a_enc_mean, a_enc_ln_var = split(self.q_a_given_x(x))
        a = F.gaussian(a_enc_mean, a_enc_ln_var)
        y_enc = self.q_y_given_a_x(a, x)
        if softmax:
            y_enc = F.softmax(y_enc)
        return y_enc

    def classification_loss(self, x, y):
        y_enc = self.predict(x, y)
        return F.softmax_cross_entropy(y_enc, y)

    def accuracy(self, x, y):
        y_enc = self.predict(x, y)
        return F.accuracy(y_enc, y)

    def __call__(self, x, y=None, y_onehot=None):
        loss = self.ELBO(x, y_onehot)
        if y is not None:
            loss += self.classification_loss(x, y)
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
