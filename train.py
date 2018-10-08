import argparse
import os
import sys

import chainer
from chainer import training
from chainer.training import extension
from chainer.training import extensions

from common.dataset import Cifar10Dataset
# from common.evaluation import calc_inception
from common.evaluation import sample_generate8
from common.record import record_setting

import gen_models.resnet_generator


def make_optimizer(model, alpha, beta1, beta2):
    optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
    optimizer.setup(model)
    return optimizer


def main():
    parser = argparse.ArgumentParser(description='Train script')
    parser.add_argument('--algorithm', '-a', type=str, default='wgan_gp_res',
                        help='GAN algorithm')
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--max_iter', type=int, default=60000)
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID')
    parser.add_argument('--out', type=str, default='result',
                        help='Directory to output the result')
    parser.add_argument('--snapshot_interval', type=int, default=10000,
                        help='Interval of snapshot')
    parser.add_argument('--evaluation_interval', type=int, default=10000,
                        help='Interval of evaluation')
    parser.add_argument('--display_interval', type=int, default=100,
                        help='Interval of display')
    parser.add_argument('--n_dis', type=int, default=5,
                        help='number of discriminator update per generator update')
    parser.add_argument('--lam', type=float, default=10,
                        help='gradient penalty')
    parser.add_argument('--adam_alpha', type=float, default=0.0002)
    parser.add_argument('--adam_beta1', type=float, default=0.0)
    parser.add_argument('--adam_beta2', type=float, default=0.9)

    args = parser.parse_args()
    record_setting(args.out)
    report_keys = ['loss_dis', 'loss_gen', 'inception_mean', 'inception_std']

    # set up  dataset
    train_dataset  = Cifar10Dataset()
    train_iter = chainer.iterators.SerialIterator(train_dataset,
                                                  args.batchsize)

    # set up netwroks and updaters
    models = []
    opts = {}
    updater_args = {"iterator": {'main': train_iter}, "device": args.gpu}

    if args.algorithm == 'wgan_gp_res':
        from updaters.wgangp_updater import Updater
        import dis_models.resnet_discriminator
        generator = gen_models.resnet_generator.ResnetGenerator()
        discriminator = dis_models.resnet_discriminator.ResnetDiscriminator()
        models = [generator, discriminator]
        report_keys.append('loss_gp')
        updater_args['n_dis'] = args.n_dis
        updater_args['lam'] = args.lam

    elif args.algorithm == 'sngan_res':
        from updaters.stdgan_updater import Updater
        import dis_models.sn_resnet_discriminator
        generator = gen_models.resnet_generator.ResnetGenerator()
        discriminator = dis_models.sn_resnet_discriminator.SNResnetDiscriminator()
        models = [generator, discriminator]
        updater_args['n_dis'] = args.n_dis
        
    elif args.algorithm == 'snwgan_res':
        from updaters.wgan_like_updater import Updater
        import dis_models.sn_resnet_discriminator
        generator = gen_models.resnet_generator.ResnetGenerator()
        discriminator = dis_models.sn_resnet_discriminator.SNResnetDiscriminator()
        models = [generator, discriminator]
        updater_args['n_dis'] = args.n_dis   
        
    else:
        raise NotImplementedError()

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        for m in models:
            m.to_gpu()

    # set up optimizers
    opts['opt_gen'] = make_optimizer(generator,
                                     args.adam_alpha,
                                     args.adam_beta1,
                                     args.adam_beta2)
    opts['opt_dis'] = make_optimizer(discriminator,
                                     args.adam_alpha,
                                     args.adam_beta1,
                                     args.adam_beta2)
    updater_args['optimizer'] = opts
    updater_args['models'] = models

    # set up updater
    updater = Updater(**updater_args)

    # set up trainer
    trainer = training.Trainer(updater, (args.max_iter, 'iteration'),
                               out=args.out)

    # set up logging
    for m in models:
        trainer.extend(extensions.snapshot_object(
            m, m.__class__.__name__ + '_{.updater.iteration}.npz'), 
            trigger=(args.snapshot_interval, 'iteration'))
    trainer.extend(extensions.LogReport(
                              keys=report_keys, 
                              trigger=(args.display_interval, 'iteration')))
    trainer.extend(extensions.PrintReport(report_keys), 
                       trigger=(args.display_interval, 'iteration'))
    trainer.extend(extensions.PlotReport(['loss_dis', 'loss_gen'], 
                                                   'iteration', 
                                                   trigger=(args.display_interval, 'iteration'), 
                                                   file_name='loss.png'), 
                       trigger=(args.display_interval, 'iteration'))
    trainer.extend(sample_generate8(generator, args.out), 
                       trigger=(args.evaluation_interval // 10, 'iteration'), 
                       priority=extension.PRIORITY_WRITER)
#     trainer.extend(calc_inception(generator, batchsize=100, n_ims=1000), 
#                        trigger=(args.evaluation_interval, 'iteration'), 
#                        priority=extension.PRIORITY_WRITER)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    
    # train
    trainer.run()


if __name__ == '__main__':
    main()
