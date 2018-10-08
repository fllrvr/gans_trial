import chainer
import numpy as np
from chainer.functions.array.broadcast import broadcast_to
from chainer.functions.connection import linear
from chainer.links.connection.linear import Linear
from common.sn.max_sv import max_singular_value

class SNLinear(Linear):
    """Linear layer with Sepctral Normalization

    Args:
        in_size (int): Dimension of input vectors.
        out_size (int): Dimension of output vectors.
        use_gamma (bool): If "True",
                        apply scalar multiplication to the normalized weight.
        nobias (bool): If "True", this function does not use the bias.
        initialW (2d array): Initial weight value.
        initial_bias (1d array): Initial bias value.
        Ip (int): The number of power iteration for calculating
                the spectral norm of W with the power iteration method.
        factor (float): A constant to adjust the spectral norm of W_bar.

    Attributes:
        W (~chiner.Variable): Weight parameter.
        W_bar (~chainer.Variabale): Spectrally normalized weight parameter.
        b (~chainer.Variable): Bias parameter.
        u (~numpy.array): Current estimation of the right singular vector
                        corresponding to the largest singular value of W.
        (optional) gamma (~chainer.Varaible): The multiple parameter.
        (optional) factor (float): A constant
                                to adjust the spectral norm of W_bar.
    """
    def __init__(self, in_size, out_size, use_gamma=False, nobias=False,
                 initialW=None, initial_bias=None, Ip=1, factor=None):
        self.Ip = Ip
        self.use_gamma = use_gamma
        self.factor = factor
        super(SNLinear, self).__init__(
            in_size, out_size, nobias, initialW, initial_bias
        )
        self.u = np.random.normal(size=(1, out_size)).astype(dtype="f")
        self.register_persistent('u')


    @property
    def W_bar(self):
        """
        Spectrally normalized weight.
        """
        sigma, _u, _ = max_singular_value(self.W, self.u, self.Ip)

        if self.factor:
            sigma = sigma / self.factor

        sigma = broadcast_to(sigma.reshape((1, 1)), self.W.shape)
        self.u[:] = _u

        if hasattr(self, 'gamma'):
            return broadcast_to(self.gamma, self.W.shape)
        else:
            return self.W / sigma


    def _initialize_params(self, in_size):
        super(SNLinear, self)._initialize_params(in_size)
        if self.use_gamma:
            _, s, _ = np.linalg.svd(self.W.data)
            with self.init_scope():
                self.gamma = chainer.Parameter(s[0], (1, 1))


    def __call__(self, x):
        if self.W.data is None:
            self._initialize_params(x.size // x.shape[0])
        return linear.linear(x, self.W_bar, self.b)
