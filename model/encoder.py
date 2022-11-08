import torch.nn as nn
import numpy as np
import torch
from model.self_ref import SelfRefLayer


def get_norm_layer(norm_type, out_filters):
    return {
        'instance': nn.InstanceNorm2d,
        'batch': nn.BatchNorm2d
    }[norm_type](out_filters)


class DownSampleBlock(nn.Module):
    def __init__(self, in_filters, out_filters, norm_type='instance'):
        super().__init__()
        self.conv = nn.Conv2d(in_filters, out_filters, 4, 2, 1)
        self.norm = get_norm_layer(norm_type, out_filters)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.leakyrelu(self.norm(self.conv(x)))


class Encoder(nn.Module):
    def __init__(
            self, nc, ndf, depth=5,
            img_shape=(32, 32),
            norm_type='instance',
            latent_dim=2048):
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim
        self.nc = nc
        self.ndf = ndf
        self.depth = depth
        self.input_conv = nn.Conv2d(nc, ndf, 1, 1, 0)

        self.conv_layers = nn.ModuleList()
        self.sf_layers = nn.ModuleList()

        ndf_cur = ndf
        for _ in range(1, self.depth):
            in_filters = ndf_cur
            ndf_cur = ndf_cur * 2
            img_shape = (tuple(int(d/2) for d in img_shape))
            self.conv_layers.append(DownSampleBlock(
                in_filters,
                ndf_cur,
                norm_type=norm_type
            ))
            self.sf_layers.append(SelfRefLayer(
                latent_dim=self.latent_dim,
                target_shape=(ndf_cur, *img_shape)
            ))
        self.output_shape = ndf_cur * np.prod(img_shape)

    def forward(self, x, z=None):
        b, _, _, _ = x.shape
        if z is None:
            z = torch.zeros(b, self.latent_dim)

        x = self.input_conv(x)
        for conv, sf in zip(self.conv_layers, self.sf_layers):
            x = conv(x)
            x = sf(x, z)
        return x.view(b, self.output_shape)
