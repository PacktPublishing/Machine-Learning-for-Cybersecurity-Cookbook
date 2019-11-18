from os.path import join, dirname
import torch
from torch import nn
from torchvision import datasets, transforms

from abs_models import utils as u
from abs_models import nets
from abs_models.inference import inference
from abs_models import sampling

DEFAULT_PATH = dirname(__file__)


class ELBOVAE(nn.Module):
    def __init__(self, AEs, n_samples, n_iter, beta, GM,
                 fraction_to_dismiss=0.1, clip=5, lr=0.05):

        super().__init__()
        self.AEs = AEs
        for i, AE in enumerate(self.AEs):
            self.add_module(f'VAE_{i}', AE)
        self.n_samples = n_samples
        self.n_iter = n_iter
        self.beta = beta
        self.GM = GM
        self.fraction_to_dismiss = fraction_to_dismiss
        self.clip = clip
        self.lr = lr
        self.logit_scale = 440
        self.confidence_level = 0.000039
        self.name_check = 'MNIST_MSE'

    def forward(self, x, return_more=False):
        # assert (torch.ge(x, 0).all())
        # assert (torch.le(x, 1).all())

        ELBOs, l_v_classes, reconsts = inference(self.AEs, x, self.n_samples, self.n_iter,
                                                 self.beta, self.GM, self.fraction_to_dismiss,
                                                 clip=self.clip, lr=self.lr)
        ELBOs = self.rescale(ELBOs)  # class specific fine-scaling

        if return_more:
            p_c = u.confidence_softmax(-ELBOs * self.logit_scale, const=self.confidence_level,
                                       dim=1)
            return p_c, ELBOs, l_v_classes, reconsts
        else:
            return -ELBOs[:, :, 0, 0]   # like logits

    def rescale(self, logits):
        return logits


class ELBOVAE_binary(ELBOVAE):
    def __init__(self, AEs, n_samples, n_iter, beta, GM,
                 fraction_to_dismiss=0.1, clip=5, lr=0.05):

        super().__init__(AEs, n_samples, n_iter, beta, GM,
                         fraction_to_dismiss=fraction_to_dismiss,
                         clip=clip, lr=lr)

        self.name_check = 'ABS'
        self.rescale_b = True
        self.discriminative_scalings = torch.tensor(
            [1., 0.96, 1.001, 1.06, 0.98, 0.96, 1.03, 1., 1., 1.]).to(u.dev())

    def forward(self, x, return_more=False):
        # assert (torch.ge(x, 0).all())
        # assert (torch.le(x, 1).all())
        x = u.binarize(x)
        return super().forward(x, return_more=return_more)

    def rescale(self, logits):
        if self.rescale_b:
            return logits * self.discriminative_scalings[None, :, None, None]
        else:
            return logits


def get_ABS(n_samples=8000, n_iter=50, beta=1, clip=5,
            fraction_to_dismiss=0.1, lr=0.05, load=True,
            binary=True, load_path=DEFAULT_PATH):
    return get_VAE(n_samples=n_samples, n_iter=n_iter, beta=beta, clip=clip,
                   fraction_to_dismiss=fraction_to_dismiss, lr=lr, load=load,
                   binary=binary, load_path=load_path)


def get_VAE(n_samples=8000, n_iter=50, beta=1, clip=5, fraction_to_dismiss=0.1, lr=0.05,
            load=True, binary=False, load_path=DEFAULT_PATH):
    """Creates the ABS model. If binary is True, returns the full
    ABS model including binarization and scalar, otherwise returns
    the base ABS model without binarization and without scalar."""

    load_path = join(DEFAULT_PATH, '../exp/VAE_swarm_MSE/nets/')

    print('ABS model')

    n_classes = 10
    nd = 8
    nx, ny = 28, 28

    def init_models():
        strides = [1, 2, 2, 1]
        latent_act_fct = u.LinearActFct

        kernelE = [5, 4, 3, 5]
        feat_mapsE = [32, 32, 64, nd]
        encoder = { 'feat_maps': feat_mapsE, 'kernels': kernelE, 'strides': strides}
        kernelD = [4, 5, 5, 4]
        feat_mapsD = [32, 16, 16, 1]
        decoder = {'feat_maps': feat_mapsD, 'kernels': kernelD, 'strides': strides}

        AEs = []
        for i in range(n_classes):
            AE = nets.VariationalAutoEncoder(encoder, decoder, latent_act_fct=latent_act_fct)
            AE.eval()
            AE.to(u.dev())
            AEs.append(AE)
        return AEs

    AEs = init_models()

    if load:
        for i in range(n_classes):
            path = load_path + f'/ABS_{i}.net'
            AEs[i].iters = 29000
            AEs[i].load_state_dict(torch.load(path, map_location=str(u.dev())))
        print('model loaded')

    GM = sampling.GaussianSamples(AEs, nd, n_classes, nx=nx, ny=ny)
    if binary:
        model = ELBOVAE_binary
    else:
        model = ELBOVAE
    model = model(AEs, n_samples, n_iter, beta, GM, fraction_to_dismiss, clip, lr=lr)
    model.eval()
    model.code_base = 'pytorch'
    model.has_grad = False
    return model


class CNN(nets.Architectures):
    def __init__(self, model):
        super().__init__()
        self.add_module('net', model)
        self.model = model
        self.has_grad = True
        self.confidence_level = 1439000
        self.logit_scale = 1
        self.name_check = 'MNIST_baseline'

    def forward(self, input):
        # assert (torch.ge(input, 0).all())
        # assert (torch.le(input, 1).all())
        return self.model.forward(input)[:, :, 0, 0]


def get_CNN(load_path=DEFAULT_PATH):

    load_path = join(DEFAULT_PATH, '../exp/mnist_cnn/nets/')


    # network
    shape = (1, 1, 28, 28)
    kernelE = [5, 4, 3, 5]
    strides = [1, 2, 2, 1]
    feat_mapsE = [20, 70, 256, 10]                   # (32, 32, 16, 2)

    model = nets.NN(feat_mapsE, shape[1:], kernels=kernelE, strides=strides)
    # load net
    print('path', load_path + '/vanilla_cnn.net')
    model.load_state_dict(torch.load(load_path + '/vanilla_cnn.net', map_location=str(u.dev())))
    print('model loaded')
    NN = CNN(model)
    NN.eval()
    NN.to(u.dev())
    NN.code_base = 'pytorch'
    return NN


class BinaryCNN(CNN):
    def __init__(self, model):
        super().__init__(model)
        self.name_check = 'MNIST_baseline_binary'

    def forward(self, input):
        input = u.binarize(input)
        return super().forward(input)


def get_binary_CNN(load_path=DEFAULT_PATH, binarize=True):
    load_path = join(DEFAULT_PATH, '../exp/mnist_cnn/nets/')

    # network
    shape = (1, 1, 28, 28)
    kernelE = [5, 4, 3, 5]
    strides = [1, 2, 2, 1]
    feat_mapsE = [20, 70, 256, 10]                   # (32, 32, 16, 2)

    model = nets.NN(feat_mapsE, shape[1:], kernels=kernelE, strides=strides)

    # load net
    model.load_state_dict(torch.load(load_path + '/vanilla_cnn.net', map_location=str(u.dev())))
    print('model loaded')
    if binarize:
        model = BinaryCNN(model)
    else:
        model = CNN(model)
    model.eval()
    model.to(u.dev())
    model.code_base = 'pytorch'
    return model


def get_transfer_model(load_path=DEFAULT_PATH):

    # new arch
    shape = (1, 1, 28, 28)
    strides = [1, 2, 2, 1]
    kernelE = [5, 4, 3, 5]
    feat_mapsE = [32, 32, 64, 10]                   # (32, 32, 16, 2)

    model = nets.NN(feat_mapsE, shape[1:], kernels=kernelE, strides=strides)
    model.load_state_dict(torch.load(load_path + 'transfer_cnn.net', map_location=str(u.dev())))

    model.to(u.dev())
    if load_path is not None:
        model.load_state_dict(torch.load(load_path, map_location=str(u.dev())))
    model.eval()
    model.code_base = 'pytorch'
    return model


class NearestNeighbor(nets.NearestNeighborLogits):
    def __init__(self, samples, classes, n_classes):
        """
        :param samples: 4D: (n_samples, nchannels, nx, ny)
        :param classes: 1D: (2, 3, 4, 1, ...) (n_samples)
        """
        super().__init__(samples, classes, n_classes)
        self.name_check = 'MNIST_NN'

    def forward(self, input_batch, return_more=False):
        # assert (torch.ge(input_batch, 0).all())
        # assert (torch.le(input_batch, 1).all())
        return super().forward(input_batch, return_more=return_more)


def get_NearestNeighbor():
    n_classes = 10
    mnist_train = datasets.MNIST('./../data', train=True, download=True,
                                 transform=transforms.Compose([transforms.ToTensor()]))

    NN = NearestNeighbor(mnist_train.train_data[:, None, ...].type(torch.float32).to(u.dev()) / 255,
                         mnist_train.train_labels.to(u.dev()), n_classes=n_classes)

    print('model initialized')
    NN.eval()           # does nothing but avoids warnings
    NN.code_base = 'pytorch'
    NN.has_grad = False
    return NN


def get_madry(load_path='./../madry/mnist_challenge/models/secret/'):
    import tensorflow as tf
    from madry.mnist_challenge.model import Model
    sess = tf.InteractiveSession()
    model = Model()
    model_file = tf.train.latest_checkpoint(load_path)
    restorer = tf.train.Saver()
    restorer.restore(sess, model_file)
    model.code_base = 'tensorflow'
    model.logit_scale = 1.
    model.confidence_level = 60.
    model.has_grad = True
    return model