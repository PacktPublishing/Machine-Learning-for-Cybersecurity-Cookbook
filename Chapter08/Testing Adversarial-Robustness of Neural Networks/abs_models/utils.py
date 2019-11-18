import torch
import torchvision
import numpy as np
import time


def get_batch(bs=1):
    loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./../data/MNIST/', train=False, download=True,
                             transform=torchvision.transforms.ToTensor()),
        batch_size=bs, shuffle=True)
    b, l = next(iter(loader))
    return t2n(b), t2n(l)


def clip_to_sphere(tens, radius, channel_dim=1):
    radi2 = torch.sum(tens**2, dim=channel_dim, keepdim=True)
    mask = torch.gt(radi2, radius**2).expand_as(tens)
    tens[mask] = torch.sqrt(
        tens[mask]**2 / radi2.expand_as(tens)[mask] * radius**2)
    return tens


def binarize(tens, thresh=0.5):
    if isinstance(tens, torch.Tensor):
        tens = tens.clone()
    else:
        tens = np.copy(tens)
    tens[tens < thresh] = 0.
    tens[tens >= thresh] = 1.
    return tens


def tens2numpy(tens):
    if tens.is_cuda:
        tens = tens.cpu()
    if tens.requires_grad:
        tens = tens.detach()
    return tens.numpy()


def t2n(tens):
    if isinstance(tens, np.ndarray):
        return tens
    elif isinstance(tens, list):
        return np.array(tens)
    elif isinstance(tens, float) or isinstance(tens, int):
        return np.array([tens])
    else:
        return tens2numpy(tens)


def n2t(tens):
    return torch.from_numpy(tens).to(dev())


class LinearActFct(torch.nn.Module):
    def forward(self, input):
        return input

    def __repr__(self):
        return self.__class__.__name__


def tsum(input, axes=None, keepdim=False):
    if axes is None:
        axes = range(len(input.size()))

    # probably some check for uniqueness of axes
    if keepdim:
        for ax in axes:
            input = input.sum(ax, keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            input = input.sum(ax, keepdim=False)

    return input


def tlog(x):
    if isinstance(x, float):
        return np.log(x)
    elif isinstance(x, int):
        return np.log(float(x))
    else:
        return torch.log(x)


def best_other(logits, gt_label):
    best_other = np.argsort(logits)
    best_other = best_other[best_other != gt_label][-1]
    return best_other


def L2(a, b, axes=None):
    if len(a.shape) != len(b.shape):
        print(a.shape, b.shape)
        raise(Exception('broadcasting not possible'))
    L2_dist = torch.sqrt(tsum((a - b)**2, axes=axes))
    return L2_dist


def auto_batch(max_batch_size, f, xs, *args, verbose=False, **kwargs):
    """Will automatically pass list subxbatches of xs to f.
    f must return torch tensors"""
    if not isinstance(xs, list):
        xs = [xs]
    n = xs[0].shape[0]
    y = []
    for start in range(0, n, max_batch_size):
        xb = [x[start:start + max_batch_size] for x in xs]
        yb = f(*xb, *args, **kwargs)
        y.append(yb)
    if not isinstance(yb, tuple):
        y = torch.cat(y)
        assert y.shape[0] == n
        return y
    else:
        return (torch.cat(y_i) for y_i in list(zip(*y)))


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed


def t_loop_collect(fct, iter_obj, *args, concat_dim=1, **kwargs):
    all_outs = []
    for obj in iter_obj:
        outs = fct(obj, *args, **kwargs)
        all_outs.append(outs)
    all_outs = list(map(list, zip(*all_outs)))
    all_outs = [torch.cat(out, dim=concat_dim) for out in all_outs]
    return all_outs

def dev():
    if torch.cuda.is_available():
        return 'cuda:0'
    else:
        return 'cpu'


def y_2_one_hot(y, n_classes=10):
    assert len(y.shape) == 1
    y_one_hot = torch.FloatTensor(y.shape[0], n_classes).to(dev())
    y_one_hot.zero_()
    return y_one_hot.scatter_(1, y[:, None], 1)


def confidence_softmax(x, const=0, dim=1):
    x = torch.exp(x)
    n_classes = x.shape[1]
    # return x
    norms = torch.sum(x, dim=dim, keepdim=True)
    return (x + const) / (norms + const * n_classes)


def cross_entropy(label, logits):
    """Calculates the cross-entropy.
    logits: np.array with shape (bs, n_classes)
    label: np.array with shape (bs)

    """
    assert label.shape[0] == logits.shape[0]
    assert len(logits.shape) == 2

    # for numerical reasons we subtract the max logit
    # (mathematically it doesn't matter!)
    # otherwise exp(logits) might become too large or too small
    logits = logits - np.max(logits, axis=1)[:, None]
    e = np.exp(logits)
    s = np.sum(e, axis=1)
    ce = np.log(s) - logits[np.arange(label.shape[0]), label]
    return ce


def show_gpu_usages(thresh=100000):
    tmp = 0
    import gc
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if obj.is_cuda and np.prod(obj.shape) > thresh \
                        and not isinstance(obj, torch.nn.parameter.Parameter):
                    tmp += 1
                    print(type(obj), list(obj.size()), obj.dtype, obj.is_cuda,
                          np.prod(obj.shape), tmp)
        except:
            pass
    print()


def binary_projection(rec, orig):
    # rec > 0.5 and orig > 0.5 --> rec[mask] = orig[mask]
    mask = [(rec >= 0.5) & (orig >= 0.5)]
    rec[mask] = orig[mask]
    # both smaller 0.5
    mask = [(rec < 0.5) & (orig < 0.5)]
    rec[mask] = orig[mask]

    # rec > 0.5 and orig < 0.5 --> rec[mask] 0.5
    rec[(rec >= 0.5) & (orig < 0.5)] = 0.5
    rec[(rec < 0.5) & (orig >= 0.5)] = 0.49999
    return rec



    pass
