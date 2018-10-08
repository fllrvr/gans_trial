import chainer
from chainer.dataset import dataset_mixin

class Cifar10Dataset(dataset_mixin.DatasetMixin):
    def __init__(self, test=False):
        d_train, d_test = chainer.datasets.get_cifar10(ndim=3, withlabel=False, scale=1.0)
        if test:
            self.ims = d_test
        else:
            self.ims = d_train
        self.ims = self.ims * 2 - 1.0
        print("load cifar-10. shape:", self.ims.shape)

    def __len__(self):
        return self.ims.shape[0]

    def get_example(self, i):
        return self.ims[i]
