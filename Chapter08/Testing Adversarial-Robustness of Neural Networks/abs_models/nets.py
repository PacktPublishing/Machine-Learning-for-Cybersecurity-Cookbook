import torch
import numpy as np
from torch import nn

from abs_models import utils as u


class Architectures(nn.Module):
    def __init__(self, input_size=None):
        super(Architectures, self).__init__()
        self.c = input_size
        self.iters = 0

    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input


class ConvAE(Architectures):
    def __init__(self, EncArgs, DecArgs):
        super().__init__(input_size=None)
        self.latent = None
        self.Encoder = ConvEncoder(**EncArgs)
        self.Decoder = ConvDecoder(**DecArgs)

    def forward(self, x):
        self.latent = self.Encoder.forward(x)
        return self.Decoder.forward(self.latent)


class VariationalAutoEncoder(ConvAE):
    def __init__(self, EncArgs, DecArgs, latent_act_fct=nn.Tanh):

        self.fac = 2

        # Decoder must match encoder
        EncArgs['feat_maps'][-1] = int(EncArgs['feat_maps'][-1] * self.fac)
        self.n_latent = int(EncArgs['feat_maps'][-1])
        self.depth = len(EncArgs['feat_maps'])

        if 'act_fcts' not in EncArgs.keys():
            EncArgs['act_fcts'] = self.depth * [torch.nn.ELU]
        EncArgs['act_fcts'][-1] = None

        # half amount of layers (half mu, half sigma)
        DecArgs['input_sizes'] = [int(EncArgs['feat_maps'][-1] / self.fac)]
        super().__init__(EncArgs, DecArgs)
        EncArgs['feat_maps'][-1] = int(EncArgs['feat_maps'][-1] / self.fac)

        self.std = None
        self.mu = None
        self.logvar = None

        self.latent_act_fct = latent_act_fct()

    def reparameterize(self, inp):
        self.mu = self.latent_act_fct(
            inp[:, :int(self.n_latent / self.fac), :, :])

        if self.training:
            # std
            self.logvar = inp[:, int(self.n_latent / 2):, :, :]
            self.std = self.logvar.mul(0.5).exp_()

            # reparam of mu
            eps = torch.empty_like(self.mu.data).normal_()
            self.latent = eps.mul(self.std).add_(self.mu)

        else:   # test
            self.latent = self.mu
            self.logvar = inp[:, int(self.n_latent / 2):, :, :]
            self.std = self.logvar.mul(0.5).exp_()

    def forward(self, x):
        prelatent = self.Encoder.forward(x)
        self.reparameterize(prelatent)
        out = self.Decoder(self.latent)
        return out


class ConvEncoder(nn.Sequential):
    def __init__(self, feat_maps=(256, 128, 128), input_sizes=(1, 28, 28),
                 kernels=(5, 3, 3),
                 BNs=None, act_fcts=None, dilations=None, strides=None):

        super().__init__()

        self.latent = None

        self.depth = len(feat_maps)
        if BNs is None:
            BNs = self.depth * [True]
            BNs[-1] = False
        if act_fcts is None:
            act_fcts = self.depth * [nn.ELU]
            act_fcts[-1] = nn.Tanh
        if dilations is None:
            dilations = self.depth * [1]
        if strides is None:
            strides = self.depth * [1]

        # check
        args = [feat_maps, kernels, dilations, strides]
        for i, it in enumerate(args):
            if len(it) != self.depth:
                raise Exception('wrong length' + str(it) + str(i))
        feat_maps = [input_sizes[0]] + list(feat_maps)

        # build net
        for i, (BN, act_fct, kx, dil, stride) in enumerate(
                zip(BNs, act_fcts, kernels, dilations, strides)):

            self.add_module('conv_%i' % i, nn.Conv2d(
                feat_maps[i], feat_maps[i + 1], kx,
                stride=stride, dilation=dil))

            if BN:
                self.add_module('bn_%i' % i, nn.BatchNorm2d(feat_maps[i + 1]))
            if act_fct is not None:
                self.add_module('nl_%i' % i, act_fct())

    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        self.latent = input
        return input


class ConvDecoder(nn.Sequential):
    def __init__(self, feat_maps=(32, 32, 1), input_sizes=(2, 1, 1),
                 kernels=(3, 3, 3),
                 BNs=None, act_fcts=None, dilations=None, strides=(1, 1, 1),
                 conv_fct=None):

        super().__init__()

        self.depth = len(feat_maps)
        if BNs is None:
            BNs = self.depth * [True]
            BNs[-1] = False
        if act_fcts is None:
            act_fcts = self.depth * [nn.ELU]
            act_fcts[-1] = u.LinearActFct
        if dilations is None:
            dilations = self.depth * [1]

        # check
        args = [feat_maps, kernels, dilations, strides]
        for i, it in enumerate(args):
            if len(it) != self.depth:
                raise Exception('wrong length' + str(it) + str(i))

        feat_maps = [input_sizes[0]] + list(feat_maps)

        if conv_fct is None:
            conv_fct = nn.ConvTranspose2d

        # build net
        for i, (BN, act_fct, kx, dil, stride) in enumerate(
                zip(BNs, act_fcts, kernels, dilations, strides)):

            self.add_module('conv_%i' % i, conv_fct(
                feat_maps[i], feat_maps[i + 1], kx, stride=stride))
            if BN:
                self.add_module('bn_%i' % i, nn.BatchNorm2d(feat_maps[i + 1]))
            self.add_module('nl_%i' % i, act_fct())


# Other models
# ------------

class NN(Architectures):
    def __init__(self, feat_maps=(16, 16, 8), input_sizes=(1, 28, 28),
                 kernels=(5, 3, 3), strides=None,
                 BNs=None, act_fcts=None):
        super().__init__(input_size=input_sizes)
        self.depth = len(feat_maps)
        ad_feat_maps = [input_sizes[0]] + list(feat_maps)

        if strides is None:
            strides = self.depth * [1]

        if BNs is None:
            BNs = self.depth * [True]
            BNs[-1] = False

        if act_fcts is None:
            act_fcts = self.depth * [nn.ELU]
            act_fcts[-1] = None

            net_builder(self, BNs, act_fcts=act_fcts, feat_maps=ad_feat_maps,
                        kernel_sizes=kernels, strides=strides)


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        bs = input.size()[0]
        return input.view((bs,) + self.shape)


class NearestNeighbor(nn.Module):
    def __init__(self, samples, classes, n_classes):
        """
        :param samples: 4D: (n_samples, nchannels, nx, ny)
        :param classes: 1D: (2, 3, 4, 1, ...) (n_samples)
        """
        super().__init__()
        self.samples = samples[None, ...]  # (1, n_samples, nch, x, y)
        self.classes = classes
        self.n_classes = n_classes
        self.max_bs = 20

    def forward(self, input_batch, return_more=True):
        assert len(input_batch.size()) == 4
        assert input_batch.size()[-1] == self.samples.size()[-1]
        assert input_batch.size()[-2] == self.samples.size()[-2]
        assert input_batch.size()[-3] == self.samples.size()[-3]

        bs = input_batch.shape[0]
        input_batch = input_batch[:, None, ...].to(u.dev())  # (bs, 1, nch, x, y)

        def calc_dist(input_batch):
            dists = u.L2(self.samples, input_batch, axes=[2, 3, 4])
            l2, best_ind_classes = torch.min(dists, 1)
            return l2, best_ind_classes

        l2s, best_ind_classes = u.auto_batch(self.max_bs, calc_dist, input_batch)

        # boring bookkeeping
        pred = self.get_classes(bs, input_batch, best_ind_classes)
        imgs = self.samples[0, best_ind_classes]
        # print(pred, imgs, l2s)\
        if return_more:
            return pred, imgs, l2s
        else:
            return pred

    def get_classes(self, bs, input_batch, best_ind_classes):
        pred = torch.zeros(bs, self.n_classes).to(u.dev())
        pred[range(bs), self.classes[best_ind_classes]] = 1.
        return pred


class NearestNeighborLogits(NearestNeighbor):
    def __init__(self, samples, classes, n_classes):
        """
        :param samples: 4D: (n_samples, nchannels, nx, ny)
        :param classes: 1D: (2, 3, 4, 1, ...) (n_samples)
        """
        super().__init__(samples, classes, n_classes=10)
        self.samples = None
        self.all_samples = samples
        self.class_samples = [self.all_samples[self.classes == i] for i in range(n_classes)]
        self.max_bs = 40

    def forward(self, input_batch, return_more=True):
        bs, nch, nx, ny = input_batch.shape
        all_imgs, all_l2s = [], []
        for i, samples in enumerate(self.class_samples):
            self.samples = samples[None, ...]
            _, imgs, l2s = super().forward(input_batch, return_more=True)
            all_imgs.append(imgs)
            all_l2s.append(l2s)

        all_l2s = torch.cat(all_l2s).view(self.n_classes, -1).transpose(0, 1)
        if return_more:
            all_imgs = torch.cat(all_imgs).view(self.n_classes, -1, nch, nx, ny).transpose(0, 1)
            return -all_l2s, all_imgs, all_l2s
        else:
            return -all_l2s

    def get_classes(self, *args, **kwargs):
        return None


def net_builder(net, BNs, act_fcts, feat_maps, kernel_sizes, strides):
    # build net
    for i, (BN, act_fct, kx, stride) in enumerate(
            zip(BNs, act_fcts, kernel_sizes, strides)):
        net.add_module('conv_%i' % i, nn.Conv2d(
            feat_maps[i], feat_maps[i + 1], kx, stride=stride))
        if BN:
            net.add_module('bn_%i' % i, nn.BatchNorm2d(feat_maps[i + 1]))
        if act_fct is not None:
            net.add_module('nl_%i' % i, act_fct())


def calc_fov(x, kernels, paddings=None, dilations=None, strides=None):
    l_x = x
    n_layer = len(kernels)
    if paddings is None:
        paddings = [0.] * n_layer
    if dilations is None:
        dilations = [1.] * n_layer
    if strides is None:
        strides = [1.] * n_layer
    for p, d, k, s in zip(paddings, dilations, kernels, strides):
        l_x = calc_fov_layer(l_x, k, p, d, s)
    return l_x


def calc_fov_layer(x, kernel, padding=0, dilation=1, stride=1):
    p, d, k, s = padding, dilation, kernel, float(stride)
    print('s', s, 'p', p, 'd', d, 'k', k, )
    if np.floor((x + 2. * p - d * (k - 1.) - 1.) / s + 1.) != (x + 2. * p - d * (k - 1.) - 1.) / s + 1.:  # noqa: E501
        print('boundary problems')
    return np.floor((x + 2. * p - d * (k - 1.) - 1.) / s + 1.)
