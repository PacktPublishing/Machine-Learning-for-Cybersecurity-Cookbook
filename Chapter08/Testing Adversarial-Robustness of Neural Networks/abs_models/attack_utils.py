import foolbox
import foolbox.attacks as fa
import numpy as np
import torch

from abs_models import utils as u
from abs_models import models


def get_attack(attack, fmodel):
    args = []
    kwargs = {}
    # L0
    if attack == 'SAPA':
        metric = foolbox.distances.L0
        A = fa.SaltAndPepperNoiseAttack(fmodel)
    elif attack == 'PA':
        metric = foolbox.distances.L0
        A = fa.PointwiseAttack(fmodel)

    # L2
    elif 'IGD' in attack:
        metric = foolbox.distances.MSE
        A = fa.L2BasicIterativeAttack(fmodel)
    elif attack == 'AGNA':
        metric = foolbox.distances.MSE
        kwargs['epsilons'] = np.linspace(0.5, 1, 50)
        A = fa.AdditiveGaussianNoiseAttack(fmodel)
    elif attack == 'BA':
        metric = foolbox.distances.MSE
        A = fa.BoundaryAttack(fmodel)
    elif 'DeepFool' in attack:
        metric = foolbox.distances.MSE
        A = fa.DeepFoolL2Attack(fmodel)
    elif attack == 'PAL2':
        metric = foolbox.distances.MSE
        A = fa.PointwiseAttack(fmodel)

    # L inf
    elif 'FGSM' in attack and not 'IFGSM' in attack:
        metric = foolbox.distances.Linf
        A = fa.FGSM(fmodel)
        kwargs['epsilons'] = 20

    elif 'IFGSM' in attack:
        metric = foolbox.distances.Linf
        A = fa.IterativeGradientSignAttack(fmodel)
    elif 'PGD' in attack:
        metric = foolbox.distances.Linf
        A = fa.LinfinityBasicIterativeAttack(fmodel)
    elif 'IGM' in attack:
        metric = foolbox.distances.Linf
        A = fa.MomentumIterativeAttack(fmodel)
    else:
        raise Exception('Not implemented')
    return A, metric, args, kwargs


class LineSearchAttack:
    def __init__(self, abs_model : models.ELBOVAE):
        self.abs = abs_model

    def __call__(self, x, l, n_coarse_steps=3, n_ft_steps=10):
        x, l = u.n2t(x), u.n2t(l)
        x, l = x.to(u.dev()), l.to(u.dev())
        bs = x.shape[0]
        best_other = 0
        best_advs = [{'original_label': -1, 'adversarial_label': None,
                      'distance': np.inf, 'img': torch.zeros(x.shape[1:]).to(u.dev())}
                     for _ in range(bs)]
        coarse_steps = torch.zeros(bs).to(u.dev())

        n_adv_found = 0
        for i, coarse_step in enumerate(torch.linspace(0, 1., n_coarse_steps).to(u.dev())):
            current_adv = (1 - coarse_step) * x + coarse_step * best_other
            best_other, current_label = self.get_best_prototypes(current_adv, l)
            for j, (current_adv_i, pred_l_i, l_i) in enumerate(zip(current_adv, current_label, l)):
                if best_advs[j]['original_label'] == -1 and pred_l_i != l_i:
                    self.update_adv(best_advs[j], current_adv_i, pred_l_i, l_i, x[j])
                    coarse_steps[i] = coarse_step
                    n_adv_found += 1
            if n_adv_found == bs:
                break
        best_advs_imgs = torch.cat([a['img'][None] for a in best_advs])
        coarse_steps_old = coarse_steps[:, None, None, None]

        # binary search
        best_advs_imgs_old = best_advs_imgs.clone()
        sign, step = - torch.ones(bs, 1, 1, 1).to(u.dev()), 0.5
        for i in range(n_ft_steps):
            coarse_steps = coarse_steps_old + step * sign
            current_adv = (1 - coarse_steps) * x + coarse_steps * best_advs_imgs_old
            _, current_label = self.get_best_prototypes(current_adv, l)

            for j, (pred_l_i, l_i) in enumerate(zip(current_label, l)):
                if pred_l_i == l_i:
                    sign[j] = 1
                else:
                    self.update_adv(best_advs[j], current_adv[j], pred_l_i, l_i, x[j])

                    sign[j] = -1
            step /= 2

        return best_advs

    def get_best_prototypes(self, x: torch.Tensor, l: torch.Tensor):
        bs = l.shape[0]
        p_c, elbos, l_v_classes, reconsts = self.abs.forward(x, return_more=True)
        _, pred_classes = torch.max(p_c, dim=1)
        p_c[range(bs), l] = - np.inf
        _, pred_classes_other = torch.max(p_c, dim=1)
        best_other_reconst = reconsts[range(bs), pred_classes_other.squeeze()]
        best_other_reconst = self.post_process_reconst(best_other_reconst, x)

        return best_other_reconst, pred_classes.squeeze()

    def update_adv(self, best_adv, current_adv, pred_l, orig_l, orig_x):
        best_adv['img'] = current_adv.data.clone()
        best_adv['original_label'] = orig_l.cpu().numpy()
        best_adv['adversarial_label'] = pred_l.cpu().numpy()
        best_adv['distance'] = np.mean((current_adv - orig_x).cpu().numpy()**2)

    def post_process_reconst(self, reconst, x):
        return reconst


class BinaryLineSearchAttack(LineSearchAttack):
    def post_process_reconst(self, reconst, x):
        return u.binary_projection(reconst, x)


def update_distal_adv(a, a_up, grads, opti):
    a_up.data = torch.from_numpy(a)
    opti.zero_grad()
    a_up.grad = torch.from_numpy(grads)
    opti.step()
    a_up.data.clamp_(0, 1)
    a = a_up.data.numpy()
    return a
