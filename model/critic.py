from torch import nn
from model.encoder import get_norm_layer
import numpy as np


def discriminator_block(
        in_filters, out_filters,
        k=4, s=2, p=1,
        normalization=True,
        norm_type='instance',
        down_sample=True):
    k, s, p = (4, 2, 1) if down_sample else (1, 1, 0)
    layers = [nn.Conv2d(in_filters, out_filters, k, stride=s, padding=p)]
    if normalization: layers.append(get_norm_layer(norm_type, out_filters))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


class Critic(nn.Module):
    def __init__(
            self,
            nc,
            ndf,
            depth=5,
            img_shape=(128, 128),
            verbose=False):
        super(Critic, self).__init__()

        self.depth = depth
        self.layers = nn.ModuleList()
        disc_block = discriminator_block(
            nc, ndf, normalization=False,
            down_sample=False)
        self.layers.append(disc_block)

        for _ in range(depth):
            ndf_cur = ndf
            ndf = ndf * 2
            img_shape = (tuple(int(d/2) for d in img_shape))
            self.layers.append(discriminator_block(ndf_cur, ndf))

        output_shape = np.prod(img_shape) * ndf
        self.fc1 = nn.Linear(output_shape, 1)

        self.verbose = verbose
        if self.verbose:
            loss_targets = [i for i in range(1, self.depth + 1)]
            print('loss_targets: ', loss_targets)

    def forward(self, x):
        b, _, _, _ = x.shape
        for layer in self.layers:
            x = layer(x)
        x = x.reshape(b, -1)
        return self.fc1(x)

    def loss(self, x, y, layer_inds=None):
        if layer_inds is None:
            layer_inds = [i for i in range(1, self.depth + 1)]

        sum = 0
        for ind, layer in enumerate(self.layers):
            x = layer(x)
            y = layer(y)
            if (ind in layer_inds) or \
                    (layer_inds is None and ind > 0):
                _, c, h, w = x.shape
                loss = ((x - y)**2).sum((1, 2, 3))
                loss = loss/(c*h*w)
                sum = sum + loss
        return sum/len(layer_inds)
