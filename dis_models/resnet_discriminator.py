import chainer
import chainer.links as L
import chainer.functions as F

def backward_linear(x_in, x, l):
    y = F.matmul(x, l.W)
    return y

def backward_convolution(x_in, x, l):
    y = F.deconvolution_2d(x, l.W, None, l.stride, l.pad, (x_in.data.shape[2], x_in.data.shape[3]))
    return y

def backward_leaky_relu(x_in, x, a):
    y = (x_in.data > 0) * x + a * (x_in.data < 0) * x
    return y

def backward_average_pooling(x_in, x, ksize=2, stride=2, pad=0):
    y = F.unpooling_2d(x, ksize, stride, pad, outsize=(x_in.data.shape[2], x_in.data.shape[3])) / 4
    return y


class OptimizedResBlock1(chainer.Chain):
    def __init__(self, ch):
        w = chainer.initializers.HeNormal()
        super(OptimizedResBlock1, self).__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(3, ch, 3, 1, 1, initialW=w)
            self.c1 = L.Convolution2D(ch, ch, 3, 1, 1, initialW=w)
            self.cs = L.Convolution2D(3, ch, 1, initialW=w)

    def __call__(self, x):
        self.h0 = x
        self.h1 = self.c0(self.h0)
        self.h2 = self.c1(F.relu(self.h1))
        self.h3 = F.average_pooling_2d(self.h2, ksize=2, stride=2)
        self.h4 = F.average_pooling_2d(self.h0, ksize=2, stride=2)
        self.h5 = self.cs(self.h4)
        self.h6 = self.h3 + self.h5
        return self.h6

    def differentiable_backward(self, g):
        gs = backward_convolution(self.h4, g, self.cs)
        gs = backward_average_pooling(self.h0, gs)
        g =  backward_average_pooling(self.h2, g)
        g = backward_convolution(self.h1, g, self.c1)
        g = backward_leaky_relu(self.h1, g, 0.0)
        g = backward_convolution(self.h0, g, self.c0)
        g = g + gs
        return g

    
class DownResBlock2(chainer.Chain):
    def __init__(self, ch):
        w = chainer.initializers.HeNormal()
        super(DownResBlock2, self).__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(ch, ch, 3, 1, 1, initialW=w) 
            self.c1 = L.Convolution2D(ch, ch, 3, 1, 1, initialW=w)
            self.cs = L.Convolution2D(ch, ch, 3, 1, 1, initialW=w)

    def __call__(self, x):
        self.h0 = x
        self.h1 = self.c0(F.relu(self.h0)) # differ from DownResBlock1
        self.h2 = self.c1(F.relu(self.h1))
        self.h3 = F.average_pooling_2d(self.h2, ksize=2, stride=2)
        self.h4 = self.cs(self.h0)
        self.h5 = F.average_pooling_2d(self.h4, ksize=2, stride=2)
        self.h6 = self.h3 + self.h5
        return self.h6

    def differentiable_backward(self, g):
        gs = backward_average_pooling(self.h4, g)
        gs = backward_convolution(self.h0, gs, self.cs)
        g = backward_average_pooling(self.h2, g)
        g = backward_convolution(self.h1, g, self.c1)
        g = backward_leaky_relu(self.h1, g, 0.0)
        g = backward_convolution(self.h0, g, self.c0)
        g = backward_leaky_relu(self.h0, g, 0.0) # differ from DownResBlock1
        g = g + gs
        return g

    
class ResBlock3(chainer.Chain):
    def __init__(self, ch):
        w = chainer.initializers.HeNormal()
        super(ResBlock3, self).__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(ch, ch, 3, 1, 1, initialW=w)
            self.c1 = L.Convolution2D(ch, ch, 3, 1, 1, initialW=w)

    def __call__(self, x):
        self.h0 = x
        self.h1 = self.c0(F.relu(self.h0))
        self.h2 = self.c1(F.relu(self.h1))
        self.h4 = self.h2 + self.h0
        return self.h4

    def differentiable_backward(self, g):
        gs = g
        g = backward_convolution(self.h1, g, self.c1)
        g = backward_leaky_relu(self.h1, g, 0.0)
        g = backward_convolution(self.h0, g, self.c0)
        g = backward_leaky_relu(self.h0, g, 0.0)
        g = g + gs
        return g

    
class ResnetDiscriminator(chainer.Chain):
    def __init__(self, bottom_width=8, ch=128, ouput_dim=1):
        w = chainer.initializers.HeNormal()
        super(ResnetDiscriminator, self).__init__()
        self.bottom_width = bottom_width
        self.ch = ch
        with self.init_scope():
            self.r0 = OptimizedResBlock1(128)
            self.r1 = DownResBlock2(128)
            self.r2 = ResBlock3(128)
            self.r3 = ResBlock3(128)
            self.l4 = L.Linear(bottom_width * bottom_width * ch, ouput_dim, initialW=w)

    def __call__(self, x):
        self.x = x
        self.h1 = self.r0(self.x)
        self.h2 = self.r1(self.h1)
        self.h3 = self.r2(self.h2)
        self.h4 = self.r3(self.h3)
        return self.l4(F.relu(self.h4))

    def differentiable_backward(self, x):
        g = backward_linear(self.h4, x, self.l4)
        g = F.reshape(g, (x.shape[0], self.ch, self.bottom_width, self.bottom_width))
        g = backward_leaky_relu(self.h4, g, 0.0)
        g = self.r3.differentiable_backward(g)
        g = self.r2.differentiable_backward(g)
        g = self.r1.differentiable_backward(g)
        g = self.r0.differentiable_backward(g)
        return g
