import numpy as np
import math

import chainer
import chainer.links as L
import chainer.functions as F

class UpResBlock(chainer.Chain):
    def __init__(self, ch):
        super(UpResBlock, self).__init__()
        with self.init_scope():
            w = chainer.initializers.HeNormal()
            # w = chainer.initializers.GlorotUniform(math.sqrt(wscale))
            self.c0 = L.Convolution2D(ch, ch, 3, 1, 1, initialW=w)
            self.c1 = L.Convolution2D(ch, ch, 3, 1, 1, initialW=w)
            self.cs = L.Convolution2D(ch, ch, 3, 1, 1, initialW=w)
            self.bn0 = L.BatchNormalization(ch)
            self.bn1 = L.BatchNormalization(ch)

    def __call__(self, x):
        h = F.relu(self.bn0(x))
        h = F.unpooling_2d(h, 2, 2, 0, cover_all=False)
        h = self.c0(h)
        h = F.relu(self.bn1(h))
        h = self.c1(h)
        hs = F.unpooling_2d(x, 2, 2, 0, cover_all=False) # shortcut part
        hs = self.cs(hs) # shortcut part
        return h + hs


class ResnetGenerator(chainer.Chain):
    def __init__(self, n_hidden=128, bottom_width=4, z_distribution="normal"):
        self.n_hidden = n_hidden
        self.bottom_width = bottom_width
        self.z_distribution = z_distribution
        super(ResnetGenerator, self).__init__()
        with self.init_scope():
            w = chainer.initializers.HeNormal()
            # w = chainer.initializers.GlorotUniform()
            self.l0 = L.Linear(n_hidden, n_hidden * bottom_width * bottom_width)
            self.r0 = UpResBlock(n_hidden)
            self.r1 = UpResBlock(n_hidden)
            self.r2 = UpResBlock(n_hidden)
            self.bn2 = L.BatchNormalization(n_hidden)
            self.c3 = L.Convolution2D(n_hidden, 3, 3, 1, 1, initialW=w)

    def make_hidden(self, batchsize):
        if self.z_distribution == "normal":
            return np.random.randn(batchsize, self.n_hidden, 1, 1,).astype(np.float32)
        elif self.z_distribution == "uniform":
            return np.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1)).astype(np.float32)
        else:
            raise Exception("unknown z distribution")

    def __call__(self, x):
        h = F.relu(self.l0(x))
        h = F.reshape(h, (x.data.shape[0], self.n_hidden, self.bottom_width, self.bottom_width))
        h = self.r0(h)
        h = self.r1(h)
        h = F.relu(self.r2(h))
        h = self.bn2(h)
        h = F.tanh(self.c3(h))
        return h
