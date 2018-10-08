import argparse
import numpy as np
import os

import chainer
import chainer.cuda
from chainer import Variable
from chainer import serializers

import sys
sys.path.append('/home/ubuntu/dl4us/gans_trial')
import gen_models.resnet_generator
from chainer_inception_score.inception_score import inception_score, Inception

def load_inception_model():
    infile = "/home/ubuntu/dl4us/gans_trial/chainer_inception_score/inception_score.model" # path to the inception model file
    model = Inception()
    serializers.load_hdf5(infile, model)
    model.to_gpu()
    return model

def main():
    parser = argparse.ArgumentParser(description='Calculate inception')
    parser.add_argument('snapshot_path', type=str)
    parser.add_argument('--batchsize', type=int, default=100)
    parser.add_argument('--n_ims', type=int, default=50000)
    args = parser.parse_args()
    
    # generatorの読み込み
    gen = gen_models.resnet_generator.ResnetGenerator()
    serializers.load_npz(args.snapshot_path, gen)
    
    # inception modelの読み込み
    model = load_inception_model()

    ims = []
    xp = gen.xp

    for i in range(0, args.n_ims, args.batchsize):
        z = Variable(xp.asarray(gen.make_hidden(args.batchsize)))
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            x = gen(z)
        x = chainer.cuda.to_cpu(x.data)
        x = np.asarray(np.clip(x * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
        ims.append(x)
        print(args.batchsize + i)
    ims = np.asarray(ims)
    _, _, _, h, w = ims.shape
    ims = ims.reshape((args.n_ims, 3, h, w)).astype("f")
    
    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        mean, std = inception_score(model, ims)

    print('Inception score mean: ', mean)
    print('Inception score std: ', std)

if __name__ == '__main__':
    main()
