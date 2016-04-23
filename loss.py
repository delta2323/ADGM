import math

from chainer.functions.activation import softplus
from chainer.functions.math import exponential
from chainer import variable


def bernoulli_nll(x, y):
    assert isinstance(x, variable.Variable)
    assert isinstance(y, variable.Variable)

    return softplus.softplus(y) - x * y


def gaussian_nll(x, mean, ln_var):
    assert isinstance(x, variable.Variable)
    assert isinstance(mean, variable.Variable)
    assert isinstance(ln_var, variable.Variable)

    x_prec = exponential.exp(-ln_var)
    x_diff = x - mean
    x_power = (x_diff * x_diff) * x_prec * -0.5
    return (ln_var + math.log(2 * math.pi)) / 2 - x_power
