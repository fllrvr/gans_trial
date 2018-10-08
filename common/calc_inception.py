import argparase
import numpy as np

import chainer
import chainer.cuda
from chainer import Variable
from chainer import serializers

import sys
sys.path.append('../')

from chainer_inception_score.inception_score import inception_score, Inception

def load_inception_model():
    infile = "../chainer_inception_score/inception_score.model"
    model = Inception()
    serializers.load_hdf5(infile, model)
    model.to_gpu()
    return model

def main():
    parser = argparse.ArgumentParser(description='Calculate inception')
    parser.add_argument('--batchsize', type=int, default=100)
    parser.add_argument('--n_ims', type=int, default=50000)
    args = parser.parse_args()

    model = load_inception_model()

    ims = []
    xp = gen.xp

    for i in range(0, args.n_ims, args.batchsize):
        z = Variable(xp.asarray(gen.make_hidden(batchsize)))
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            x = gen(z)
        x = chainer.cuda.to_cpu(x.data)
        x = np.asarray(np.clip(x * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
        ims.append(x)
        print(batchsize + i)
    ims = np.asarray(ims)
    _, _, _, h, w = ims.shape
    ims = ims.reshape((n_ims, 3, h, w)).astype("f")
    print(ims.shape)

    mean, std = inception_score(model, ims)

    return mean, std

if __name__ == '__main__':
    main()
