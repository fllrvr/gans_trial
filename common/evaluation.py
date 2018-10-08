import os
import sys

import numpy as np
from PIL import Image

import chainer
import chainer.cuda
from chainer import Variable
from chainer import serializers

sys.path.append(os.path.dirname(__file__))
from chainer_inception_score.inception_score import inception_score, Inception


def load_inception_model():
    infile = "chainer_inception_score/inception_score.model"
    model = Inception()
    serializers.load_hdf5(infile, model)
    model.to_gpu()
    return model


def calc_inception(gen, batchsize=100, n_ims=50000):
    @chainer.training.make_extension()
    def evaluation(trainer):
        model = load_inception_model()

        ims = []
        xp = gen.xp

        for i in range(0, n_ims, batchsize):
            z = Variable(xp.asarray(gen.make_hidden(batchsize)))
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                x = gen(z)
            x = chainer.cuda.to_cpu(x.data)
            x = np.asarray(np.clip(x * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
            ims.append(x)
        ims = np.asarray(ims)
        _, _, _, h, w = ims.shape
        ims = ims.reshape((n_ims, 3, h, w)).astype("f")

        mean, std = inception_score(model, ims)

        chainer.reporter.report({'inception_mean': mean, 'inception_std': std})

    return evaluation


def sample_generate5(gen, dst, seed=0):
    @chainer.training.make_extension()
    def make_image(trainer):
        np.random.seed(seed)
        n_images = 5*5
        xp = gen.xp
        z = Variable(xp.asarray(gen.make_hidden(n_images)))
        with chainer.using_config('train', False), chainer.using_config('enabale_backprop', False):
            x = gen(z)
        x = chainer.cuda.to_cpu(x.data)
        np.random.seed()

        x = np.asarray(np.clip(x * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
        _, _, H, W = x.shape
        x = x.reshape((5, 5, 3, H, W)) # (rows, cols, channel, H, W)
        x = x.transpose(0, 3, 1, 4, 2)
        x = x.reshape((5 * H, 5 * W, 3))

        preview_dir = '{}/preview'.format(dst)
        preview_path = preview_dir + '/image{:0>8}.png'.format(trainer.updater.iteration)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        Image.fromarray(x).save(preview_path)

    return make_image

def sample_generate8(gen, dst, seed=0):
    @chainer.training.make_extension()
    def make_image(trainer):
        np.random.seed(seed)
        n_images = 8*8
        xp = gen.xp
        z = Variable(xp.asarray(gen.make_hidden(n_images)))
        with chainer.using_config('train', False), chainer.using_config('enabale_backprop', False):
            x = gen(z)
        x = chainer.cuda.to_cpu(x.data)
        np.random.seed()

        x = np.asarray(np.clip(x * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
        _, _, H, W = x.shape
        x = x.reshape((8, 8, 3, H, W)) # (rows, cols, channel, H, W)
        x = x.transpose(0, 3, 1, 4, 2)
        x = x.reshape((8 * H, 8 * W, 3))

        preview_dir = '{}/preview'.format(dst)
        preview_path = preview_dir + '/image{:0>8}.png'.format(trainer.updater.iteration)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        Image.fromarray(x).save(preview_path)

    return make_image
