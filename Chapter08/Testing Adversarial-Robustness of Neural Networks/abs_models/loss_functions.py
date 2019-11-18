import torch
from abs_models import utils as u
import numpy as np

def squared_L2_loss(a, b, axes, keepdim=True):
    return u.tsum((a - b)**2, axes=axes, keepdim=keepdim)


def KLD(mu_latent_q, sig_q=1., dim=-3):
    """

    :param mu_latent_q: z must be shape (..., n_latent ...) at i-th pos
    :param sig_q:  scalar
    :param dim: determines pos i
    :return:
    """
    return -0.5 * torch.sum(1 - mu_latent_q ** 2 + u.tlog(sig_q) - sig_q**2,
                            dim=dim, keepdim=True)


def ELBOs(x_rec: torch.Tensor, samples_latent: torch.Tensor, x_orig: torch.Tensor,
          beta=1, dist_fct=squared_L2_loss):
    """
    :param x_rec: shape (..., n_channels, nx, ny)
    :param samples_latent:  (..., n_latent, 1, 1)
    :param x_orig:  (..., n_channels, nx, ny)
    :param beta:
    :param dist_fct:
    :return:
    """
    n_ch, nx, ny = x_rec.shape[-3:]
    kld = KLD(samples_latent, sig_q=1.)
    rec_loss = dist_fct(x_orig, x_rec, axes=[-1, -2, -3])
    elbo = rec_loss + beta * kld
    # del x_rec, x_orig, kld
    # del x_rec, samples_latent, x_orig
    return elbo / (n_ch * nx * ny)


def ELBOs2(x, rec_x, samples_latent, beta):
    """This is the loss function used during inference to calculate the logits.

    This function must only operate on the last the dimensions of x and rec_x.
    There can be varying number of additional dimensions before them!
    """

    input_size = int(np.prod(x.shape[-3:]))
    assert len(x.shape) == 4 and len(rec_x.shape) == 4
    # alternative implementation that is much faster and more memory efficient
    # when each sample in x needs to be compared to each sample in rec_x
    assert x.shape[-3:] == rec_x.shape[-3:]
    x = x.reshape(x.shape[0], input_size)
    y = rec_x.reshape(rec_x.shape[0], input_size)

    x2 = torch.norm(x, p=2, dim=-1, keepdim=True).pow(2)  # x2 shape (bs, 1)
    y2 = torch.norm(y, p=2, dim=-1, keepdim=True).pow(2)  # y2 shape (1, nsamples)
    # note that we could cache the calculation of y2, but
    # it's so fast that it doesn't matter

    L2squared = x2 + y2.t() - 2 * torch.mm(x, y.t())
    L2squared = L2squared / input_size

    kld = KLD(samples_latent, sig_q=1.)[None, :, 0, 0, 0] / input_size
    # note that the KLD sum is over the latents, not over the input size
    return L2squared + beta * kld


