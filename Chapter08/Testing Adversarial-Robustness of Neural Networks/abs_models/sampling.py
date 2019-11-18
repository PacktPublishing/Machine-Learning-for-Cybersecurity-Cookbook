import torch
from abs_models import utils as u
from torch import tensor
import numpy as np
from torch.nn import functional as F
from scipy.stats import multivariate_normal


class GridMan(object):
    def __init__(self, AEs, nd, n_classes, nc=1, nx=28, ny=28, limit=0.99):
        self.samples = {}
        self.images = {}
        self.th_images = {}
        self.classes = {}
        self.l_v = {}
        self.AEs = AEs
        self.nd = nd
        self.n_classes = n_classes
        self.nx = nx
        self.ny = ny
        self.nc = nc
        self.limit = None

    def init_grid(self, n_samples, fraction_to_dismiss=None,
                  sample_sigma=None):
        n_grid = self.n_samples_to_n_grid(n_samples)
        print('init new grid', n_samples, n_grid)
        limit = 0.99
        if self.limit is not None:
            limit = self.limit
        grids = [(np.linspace(-limit, limit, n_grid)) for i in range(self.nd)]
        xys = np.array(np.meshgrid(*grids))
        xys = np.moveaxis(xys, 0, -1).reshape(n_grid ** self.nd, self.nd)
        self.samples[n_samples] = xys
        self.l_v[n_samples] = \
            torch.from_numpy(xys[:, :, None, None].astype(np.float32)).to(u.dev())

    def get_images(self, n_samples=10, fraction_to_dismiss=0.1,
                   weighted=False, sample_sigma=1):
        if n_samples not in self.images.keys():
            self.init_grid(n_samples, fraction_to_dismiss=fraction_to_dismiss,
                           sample_sigma=sample_sigma)
            self.images[n_samples] = np.empty((self.n_classes, n_samples,
                                               self.nc, self.nx, self.ny))
            for c, AE in enumerate(self.AEs[:self.n_classes]):
                AE.eval()
                images = torch.sigmoid(AE.Decoder.forward(self.l_v[n_samples])).cpu().data.numpy()
                if weighted:
                    images = images[:, 0, None]
                self.images[n_samples][c, ...] = images

            self.l_v[n_samples]
            assert n_samples not in self.th_images
            self.th_images[n_samples] = tensor(self.images[n_samples]).type(
                torch.FloatTensor).to(u.dev())
            print('done creating samples')

        return self.images[n_samples]

    def n_samples_to_n_grid(self, n_samples):
        return int(np.round(n_samples ** (1. / self.nd)))


class GaussianSamples(GridMan):
    def init_grid(self, n_samples, fraction_to_dismiss=0.1,
                  mus=None, sample_sigma=1):
        if mus is None:
            mus = np.zeros(self.nd)
        samples = get_gaussian_samples(n_samples, self.nd, mus,
                                       fraction_to_dismiss=fraction_to_dismiss,
                                       sample_sigma=sample_sigma)
        self.samples[n_samples] = samples
        self.l_v[n_samples] = \
            torch.from_numpy(samples[:, :, None, None].astype(
                np.float32)).to(u.dev())

    def n_samples_to_n_grid(self, n_samples):
        return n_samples


def get_gaussian_samples(n_samples, nd, mus,
                         fraction_to_dismiss=0.1, sample_sigma=1):
    # returns nd coords sampled from gaussian in shape (n_samples, nd)
    sigmas = np.diag(np.ones(nd)) * sample_sigma
    g = multivariate_normal(mus, sigmas)
    samples = g.rvs(size=int(n_samples / (1. - fraction_to_dismiss)))
    probs = g.pdf(samples)
    thresh = np.sort(probs)[-n_samples]
    samples = samples[probs >= thresh]
    return samples
