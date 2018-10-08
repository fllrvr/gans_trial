import chainer
from chainer import cuda
import chainer.functions as F


def _l2normalize(v, eps=1e-12):
    return v / (((v ** 2).sum()) ** 0.5 + eps)


def max_singular_value(W, u=None, Ip=1):
    """
    Apply power iteration for the weight parameter
    to compute the max sigular value.
    """
    if not Ip >= 1:
        raise ValueError("The number of power iteration should be positive integer")

    xp = cuda.get_array_module(W.data)
    if u is None:
        u = xp.random.normal(size=(1, W.shape[0]).astype(xp.float32))
    _u = u
    for _ in range(Ip):
        _v = _l2normalize(xp.dot(_u, W.data), eps=1e-12)
        _u = _l2normalize(xp.dot(_v, W.data.transpose()), eps=1e-12)
    sigma = F.sum(F.linear(_u, F.transpose(W)) * _v)
    return sigma, _u, _v
