import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image

from abs_models import utils as u


def visualize_image(ax, im, title=None, clear=False, **kwargs):
    if clear:
        ax.cla()
    ax.imshow(im, **kwargs)
    if title is not None:
        ax.set_title(title)
    ax.axis('off')
    return(ax)


def plot(ax, y_datas, x_data=None, title=None, clear=True,
         scale=None, legend=None):
    if not any(isinstance(i, list) for i in y_datas):
        y_datas = [y_datas]
    if clear:
        ax.clear()
    if x_data is None:
        x_data = range(len(y_datas[0]))

    # acutal plotting
    plots = []
    for y_data in y_datas:
        pl, = ax.plot(x_data, y_data)
        plots.append(pl)

    if legend:
        ax.legend(plots, legend)
    if scale is not None:
        ax.set_yscale(scale)
    if title is not None:
        ax.set_title(title)
    return ax


def scatter(ax, x_data, y_data, title=None, clear=True):
    if clear:
        ax.clear()
    ax.scatter(x_data, y_data)
    if title is not None:
        ax.set_title(title)


def subplots(*args, height=6, width=15, **kwargs):
    fig, ax = plt.subplots(*args, squeeze=False, **kwargs)
    if height is not None:
        fig.set_figheight(height)
    if width is not None:
        fig.set_figwidth(width)
    return fig, ax


class Visualizer:
    def __init__(self):
        self.plots = {}
        self.i = -1
        self.reset()

    def reset(self):
        self.ny = 4
        self.nx = 4
        fig = plt.figure()
        plt.ion()
        fig.show()
        fig.canvas.draw()

        self.fig = fig
        self.i = 0
        # for key in self.plots.keys():
        #     self.plots[key].ax = self.get_next_plot()

    def add_scalar(self, name, y, x):
        y = u.t2n(y)
        if name in self.plots.keys():
            self.plots[name].x.append(x)
            self.plots[name].y.append(y)
        else:
            self.plots[name] = PlotObj(x, y, self.get_next_plot())
        self.plots[name].ax.clear()
        plot(self.plots[name].ax, self.plots[name].y,
             self.plots[name].x, title=name)
        self.fig.canvas.draw()

    def add_image(self, name, img, x):
        if not isinstance(img, np.ndarray):
            img = u.t2n(img)
        img = img.squeeze()
        if name not in self.plots.keys():
            self.plots[name] = self.plots[name] \
                = PlotObj(0, 0, self.get_next_plot())
        visualize_image(self.plots[name].ax, img, title=name, cmap='gray')

    def get_next_plot(self):
        self.i += 1
        ax = self.fig.add_subplot(self.nx, self.ny, self.i)
        return ax


class PlotObj:
    def __init__(self, x, y, ax):
        self.x = [x]
        self.y = [y]
        self.ax = ax


# visualize hidden space
class RobNNVisualisor(object):
    def __init__(self):
        self.xl = []
        self.yl = []
        self.cl = []

    def generate_data(self, model, loader, cuda=False):
        for i, (test_data, test_label) in enumerate(loader):
            if i == int(np.ceil(400 / loader.batch_size)):
                break
            x = test_data
            yt = test_label
            x = x.to(u.dev())
            model.forward(x)
            latent = model.latent.cpu().data.numpy().swapaxes(0, 1).squeeze()
            self.xl += latent[0].tolist()
            self.yl += latent[1].tolist()
            self.cl += yt.data.numpy().tolist()

    def visualize_hidden_space(self, fig, ax, model=None,
                               loader=None, cuda=False,
                               reload=False, colorbar=False):
        if self.xl == [] or reload:
            self.generate_data(model, loader, cuda=cuda)
        cmap = plt.cm.get_cmap("viridis", 10)

        pl = ax.scatter(self.xl, self.yl, c=self.cl, label=self.cl,
                        vmin=-0.5, vmax=9.5, cmap=cmap)

        if colorbar:
            fig.colorbar(pl, ax=ax, ticks=range(10))
        return ax


def fig2img(fig):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format
    and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data(fig)
    w, h, d = buf.shape
    return Image.frombytes("RGBA", (w, h), buf.tostring())


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with
    RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode.
    # Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


# adapted from https://github.com/lanpa/tensorboard-pytorch
def tens2scattters(tens, lims=None, labels=None):
    tens_np = u.tens2numpy(tens)
    labels = u.tens2numpy(labels)

    # draw
    fig = plt.figure()
    ax = plt.gca()
    ax.scatter(tens_np[0], tens_np[1], c=labels)
    plt.axis('scaled')
    if lims is not None:
        ax.set_xlim(lims[0], lims[1])
        ax.set_ylim(lims[0], lims[1])
    return fig2data(fig)


def fig2data(fig):
    fig.canvas.draw()
    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def visualize_latent_distr(CNN, nd, limit=2, n_grid=100):
    limit = 2
    n_grid = 100
    fig, ax = subplots(1, 1, width=7, height=6)
    fig.subplots_adjust(right=0.8)
    grids = [(np.linspace(-limit, limit, n_grid)) for i in range(nd)]
    xys = np.array(np.meshgrid(*grids))
    xys = np.moveaxis(xys, 0, -1).reshape(n_grid ** nd, nd)
    outs = CNN.forward(torch.from_numpy(xys[:, :, None, None]).type(torch.cuda.FloatTensor))  # noqa: E501
    outs = u.t2n(outs.squeeze())
    sc = ax[0, 0].scatter(xys[:, 0], xys[:, 1], c=(outs - np.min(outs)) / (np.max(outs) - np.min(outs)))  # noqa: E501
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(sc, cax=cbar_ax)
    return fig2data(fig)


if __name__ == '__main__':
    fig, ax = subplots(2)
    print(ax)
