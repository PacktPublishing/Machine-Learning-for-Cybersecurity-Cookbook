import torch
from torch import optim
from torch.nn import functional as F
from torch import tensor
import numpy as np

from abs_models import utils as u
from abs_models import loss_functions


def inference(AEs, x_inp, n_samples, n_iter, beta, GM, fraction_to_dismiss=0.1, lr=0.01,
              n_classes=10, nd=8, clip=2, GD_inference_b=True,
              dist_fct=loss_functions.squared_L2_loss):

    if n_iter == 0:
        GD_inference_b = False

    tmp_bs = x_inp.size()[0]

    # get_images has built-in caching
    if n_samples not in GM.th_images:
        print('setting random seed')
        # fix random numbers for attacks
        torch.cuda.manual_seed_all(999)
        torch.manual_seed(1234)
        np.random.seed(1234)
        # generate a bunch of samples for each VAE
        GM.get_images(n_samples, fraction_to_dismiss)

    # use caching for conversion to torch
    x_test_samples = GM.th_images[n_samples]

    # calculate the likelihood for all samples
    with torch.no_grad():
        bs, n_ch, nx, ny = x_inp.shape
        n_samples, n_latent = GM.l_v[n_samples].shape[-4:-2]

        all_ELBOs = \
            [loss_functions.ELBOs2(x_inp, recs.detach(), GM.l_v[n_samples], beta)
             for recs in x_test_samples]
        all_ELBOs = torch.stack(all_ELBOs, dim=1)

    x_inp = x_inp.view(bs, n_ch, nx, ny)

    # tmp save memory
    # GM.th_images[n_samples] = GM.th_images[n_samples].cpu()
    # GM.l_v[n_samples] = GM.l_v[n_samples].cpu()

    # select the best prototype for each VAE
    min_val_c, min_val_c_args = torch.min(all_ELBOs, dim=2)
    indices = min_val_c_args.view(tmp_bs * n_classes)
    # l_v_best shape: (bs, n_classes, 8, 1, 1)
    l_v_best = GM.l_v[n_samples][indices].view(tmp_bs, n_classes, nd, 1, 1)

    if GD_inference_b:  # gradient descent in latent space
        return GD_inference(AEs, l_v_best.data, x_inp.data,
                            clip=clip, lr=lr, n_iter=n_iter, beta=beta, dist_fct=dist_fct)
    else:
        if tmp_bs == 1:
            all_recs = GM.images[n_samples][list(range(n_classes)), u.t2n(indices), :, :, :]
        else:
            all_recs = None
        return min_val_c, l_v_best, all_recs


def GD_inference(AEs, l_v_best, x_inp, clip=5, lr=0.01, n_iter=20,
                 beta=1, dist_fct=loss_functions.squared_L2_loss):
    n_classes = len(AEs)

    # l_v_best are the latents
    # has shape (batch_size, n_classes == 10, n_latents == 8) + singleton dims

    # do gradient descent w.r.t. ELBO in latent space starting from l_v_best
    def gd_inference_b(l_v_best, x_inp, AEs, n_classes=10, clip=5, lr=0.01, n_iter=20,
                       beta=1, dist_fct=loss_functions.squared_L2_loss):

        bs, n_ch, nx, ny = x_inp.shape
        with torch.enable_grad():
            l_v_best = l_v_best.data.clone().detach().requires_grad_(True).to(u.dev())
            opti = optim.Adam([l_v_best], lr=lr)
            for i in range(n_iter):
                ELBOs = []
                all_recs = []
                for j in range(n_classes):
                    if i == n_iter - 1:
                        l_v_best = l_v_best.detach()  # no gradients in last run
                    AEs[j].eval()

                    rec = torch.sigmoid(AEs[j].Decoder.forward(l_v_best[:, j]))

                    ELBOs.append(loss_functions.ELBOs(rec,              # (bs, n_ch, nx, ny)
                                                      l_v_best[:, j],   # (bs, n_latent, 1, 1)
                                                      x_inp,            # (bs, n_ch, nx, ny)
                                                      beta=beta,
                                                      dist_fct=dist_fct))
                    if i == n_iter - 1:
                        all_recs.append(rec.view(bs, 1, n_ch, nx, ny).detach())

                ELBOs = torch.cat(ELBOs, dim=1)
                if i < n_iter - 1:
                    loss = (torch.sum(ELBOs)) - 8./784./2  # historic reasons
                    # backward
                    opti.zero_grad()
                    loss.backward()
                    opti.step()
                    l_v_best.data = u.clip_to_sphere(l_v_best.data, clip, channel_dim=2)
                else:
                    opti.zero_grad()
                    all_recs = torch.cat(all_recs, dim=1)

        return ELBOs.detach(), l_v_best.detach(), all_recs

    ELBOs, l_v_best, all_recs = u.auto_batch(1000, gd_inference_b, [l_v_best, x_inp], AEs,
                                             n_classes=n_classes, clip=clip, lr=lr,
                                             n_iter=n_iter, beta=beta, dist_fct=dist_fct)

    return ELBOs, l_v_best, all_recs


# pytorch 1.0:
# def GD_inference_new(AEs, l_v_best, x_inp, clip=5, lr=0.01, n_iter=20,
    #              beta=1, dist_fct=loss_functions.squared_L2_loss):
    # n_classes = len(AEs)
    #
    # # l_v_best are the latents
    # # have shape (batch_size, n_classes == 10, n_latents == 8) + singleton dims
    #
    # # do gradient descent w.r.t. ELBO in latent space starting from l_v_best
    # def gd_inference_b(l_v_best, x_inp, AEs, clip=5, lr=0.01, n_iter=20,
    #                    beta=1, dist_fct=loss_functions.squared_L2_loss):
    #
    #     with torch.enable_grad():
    #         l_v_best = l_v_best.data.clone().detach().requires_grad_(True).to(u.dev())
    #         opti = optim.Adam([l_v_best], lr=lr)
    #         for i in range(n_iter):
    #             recs = torch.nn.parallel.parallel_apply(
    #                 [AE.Decoder.forward for AE in AEs],
    #                 [best_latent for best_latent in l_v_best.transpose(0, 1)])
    #             recs = torch.nn.functional.sigmoid(torch.stack(recs, dim=1))
    #             ELBOs = loss_functions.ELBOs(recs, l_v_best, x_inp[:, None], beta=beta,
    #                                          dist_fct=dist_fct)[..., 0]
    #
    #             if i < n_iter - 1:
    #                 loss = (torch.sum(ELBOs)) - 8./784./2  # historic reasons
    #                 # backward
    #                 opti.zero_grad()
    #                 loss.backward()
    #                 opti.step()
    #                 l_v_best.data = u.clip_to_sphere(l_v_best.data, clip, channel_dim=2)
    #             else:
    #                 opti.zero_grad()
    #
    #     return ELBOs.detach(), l_v_best.detach(), recs.detach()
    #
    # ELBOs, l_v_best, all_recs = u.auto_batch(2000, gd_inference_b, [l_v_best, x_inp], AEs,
    #                                          clip=clip, lr=lr,  n_iter=n_iter, beta=beta, dist_fct=dist_fct)
    #
    # return ELBOs, l_v_best, all_recs
