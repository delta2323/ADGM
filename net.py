import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np
import six


class ADGM(chainer.Chain):

    h_dim = 10

    def __init__(self, x_dim, a_dim, y_dim, z_dim):
        q_a_given_x = MLP((x_dim, h_dim, a_dim))
        q_z_given_a_y_x = MultiMLP((a_dim, y_dim, x_dim), (h_dim, z_dim))
        q_y_given_a_x = MultiMLP((a_dim, x_dim), (h_dim, y_dim))
        p_a_given_z_y_x = MultiMLP((z_dim, y_dim, x_dim), (h_dim, a_dim))
        p_x_given_z_y = MultiMLP((z_dim, y_dim), (h_dim, z_dim))
        super(ADGM, self).__init__(
            q_a_given_x=q_a_given_x,
            q_z_given_a_y_x=q_z_given_a_y_x,
            q_y_given_a_x=q_y_given_a_x,
            p_a_given_z_y_x=p_a_given_z_y_x,
            p_x_given_z_y=p_x_given_z_y)

    def _split(self, x):
        return F.split_axis(x, 2, 1)

    @property
    def train(self, val):
        self.q_a_given_x = val
        self.q_z_given_a_y_x = val
        self.q_y_given_a_x = val
        self.p_a_given_z_y_x = val
        self.p_x_given_z_y = val

    def ELBO(self, x, y=None):
        a_enc_mean, a_enc_ln_var = self._split(self.q_a_given_x(x))
        a = F.gaussian(a_enc_mean, a_enc_ln_var)
        if y is None:
            y = F.softmax(self.q_y_given_a_x(a, x))
        z_enc_mean, z_enc_ln_var = self._split(self.q_z_given_a_y_x(a, y, x))
        z = F.gaussian(z_enc_mean, z_enc_ln_var)

        a_dec_mean, a_dec_ln_var = self.p_a_given_z_y_x(z, y, x)
        x_dec_mean, x_dec_ln_var = self.p_x_given_z_y(z, y)

        nll_p_z = F.gaussian_nll(z, 0, 0)
        nll_p_x_given_z_y = F.gaussian_nll(x, x_dec_mean, x_dec_ln_var)
        nll_p_a_given_z_y_x = F.gaussian_nll(a, a_dec_mean, a_dec_ln_var)
        nll_q_a_given_x = F.gaussian_nll(a, a_enc_mean, a_enc_ln_var)
        nll_q_z_given_a_y_x = F.gaussian_nll(z, z_enc_mean, z_enc_ln_var)

        if y is None:
            nll_q_y_given_a_x = -entropy(y)
        else:
            nll_q_y_given_a_x = 0.0

        return F.sum(nll_p_z + nll_p_x_given_z_y + nll_p_a_given_z_y_x +
                     nll_q_y_given_a_x + nll_q_a_given_x + nll_q_z_given_a_y_x)
        
    def classification(self, x, y):
        a_enc_mean, a_enc_ln_var = self._split(self.q_a_given_x(x))
        a = F.gaussian(a_enc_mean, a_enc_ln_var)
        y_enc = self.q_y_given_a_x(a, x)
        return F.softmax_cross_entropy(y_enc, y)

    def __call__(self, x, y=None):
        loss = self.ELBO(x, y)
        if y is not None:
            loss += self.classification(x, y)
        self.loss = float(loss.data)
        return loss


class MultiMLP(chainer.Chain):

    def __init__(self, input_dims, units):
        assert len(units) > 0
        encoders = chainer.ChainList([
                chainer.Linear(input_dim, units[0])
                for input_dim in six.moves.range(input_dims)])
        super(self, MultiMLP).init(
            encoders=encoders, mlps=MLP(units),
            bn=chainer.links.BatchNormalization(units[0]))
        self.train = True

    def __call__(self, *xs):
        assert len(xs) == len(self.encoders)
        xs = [self.encoders(x) for x in xs]
        x = self.relu(self.bn(F.sum(xs)))
        return self.encoder(x)
        

class MLP(chainer.Chain):

    def __init__(self, units):
        self.layer_num = len(units) - 1
        linear = chainer.ChainList([
                chainer.Linear(units[i], units[i+1])
                for i in six.moves.range(self.layer_num)])
        bn = chainer.ChainList(
            chainer.links.BatchNormalization(units[i])
            for i in six.moves.range(self.layer_num - 1))
        super(MLP, self).__init__(linear=linear, bn=bn)
        self.train = True

    def __call__(self, x):
        for i in six.moves.range(self.layer_num - 1):
            x = self.linear[i](x)
            x = self.bn[i](x, test=not self.train)
            x = F.relu(x)
        return self.linear[self.layer_num - 1]
