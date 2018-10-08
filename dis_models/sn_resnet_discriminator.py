import numpy as np

import chainer
import chainer.functions as F
from common.sn.sn_linear import SNLinear
from common.sn.sn_convolution_2d import SNConvolution2D
from chainer.functions.array.broadcast import broadcast_to

class SNOptimizedResBlock1(chainer.Chain):
    def __init__(self, ch):
        w = chainer.initializers.HeNormal()
        super(SNOptimizedResBlock1, self).__init__()
        with self.init_scope():
            self.c0 = SNConvolution2D(3, ch, 3, 1, 1, initialW=w)
            self.c1 = SNConvolution2D(ch, ch, 3, 1, 1, initialW=w)
            self.cs = SNConvolution2D(3, ch, 1, initialW=w)

    def __call__(self, x):
        self.h0 = x
        self.h1 = self.c0(self.h0)
        self.h2 = self.c1(F.relu(self.h1))
        self.h3 = F.average_pooling_2d(self.h2, ksize=2, stride=2) * 2
        self.h4 = F.average_pooling_2d(self.h0, ksize=2, stride=2) * 2
        self.h5 = self.cs(self.h4)
        self.h6 = (self.h3 + self.h5) / np.sqrt(2)
        return self.h6

    
class SNDownResBlock2(chainer.Chain):
    def __init__(self, ch):
        w = chainer.initializers.HeNormal()
        super(SNDownResBlock2, self).__init__()
        with self.init_scope():
            self.c0 = SNConvolution2D(ch, ch, 3, 1, 1, initialW=w) # differ from DownResBlock1
            self.c1 = SNConvolution2D(ch, ch, 3, 1, 1, initialW=w)
            self.cs = SNConvolution2D(ch, ch, 3, 1, 1, initialW=w) # differ from DownResBlock1

    def __call__(self, x):
        self.h0 = x
        self.h1 = self.c0(F.relu(self.h0)) # differ from DownResBlock1
        self.h2 = self.c1(F.relu(self.h1))
        self.h3 = F.average_pooling_2d(self.h2, ksize=2, stride=2) * 2
        self.h4 = self.cs(self.h0)
        self.h5 = F.average_pooling_2d(self.h4, ksize=2, stride=2) * 2
        self.h6 = (self.h3 + self.h5) / np.sqrt(2)
        return self.h6

    
class SNResBlock3(chainer.Chain):
    def __init__(self, ch):
        w = chainer.initializers.HeNormal()
        super(SNResBlock3, self).__init__()
        with self.init_scope():
            self.c0 = SNConvolution2D(ch, ch, 3, 1, 1, initialW=w)
            self.c1 = SNConvolution2D(ch, ch, 3, 1, 1, initialW=w)

    def __call__(self, x):
        self.h0 = x
        self.h1 = self.c0(F.relu(self.h0))
        self.h2 = self.c1(F.relu(self.h1))
        self.h4 = self.h2 + self.h0
        self.h5 = self.h4 / np.sqrt(2) # normalize
        return self.h5

    
class SNResnetDiscriminator(chainer.Chain):
    def __init__(self, bottom_width=8, ch=128, ouput_dim=1):
        w = chainer.initializers.HeNormal()
        super(SNResnetDiscriminator, self).__init__()
        self.bottom_width = bottom_width
        self.ch = ch
        with self.init_scope():
            self.r0 = SNOptimizedResBlock1(128)
            self.r1 = SNDownResBlock2(128)
            self.r2 = SNResBlock3(128)
            self.r3 = SNResBlock3(128)
            self.l4 = SNLinear(bottom_width * bottom_width * ch, ouput_dim, initialW=w)

    def __call__(self, x):
        self.x = x
        self.h1 = self.r0(self.x)
        self.h2 = self.r1(self.h1)
        self.h3 = self.r2(self.h2)
        self.h4 = self.r3(self.h3)
        return self.l4(F.relu(self.h4))
