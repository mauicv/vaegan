import torch.nn as nn
import numpy as np
from model.resnet import ResnetBlock


class DownSampleInstanceConv2dBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super().__init__()
        self.conv = nn.Conv2d(in_filters, out_filters, 4, 2, 1)
        self.norm = nn.InstanceNorm2d(out_filters)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.leakyrelu(self.norm(self.conv(x)))


class DownSampleBatchConv2dBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super().__init__()
        self.conv = nn.Conv2d(in_filters, out_filters, 4, 2, 1)
        self.norm = nn.BatchNorm2d(out_filters)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.leakyrelu(self.norm(self.conv(x)))


class Encoder(nn.Module):
    def __init__(
            self, nc, ndf, depth=5,
            img_shape=(32, 32),
            res_blocks=tuple(0 for _ in range(5)),
            downsample_block_type=DownSampleInstanceConv2dBlock
        ):
        super(Encoder, self).__init__()

        assert len(res_blocks) == depth, 'len(res_blocks) != depth'

        self.nc = nc
        self.ndf = ndf
        self.depth = depth
        self.input_conv = nn.Conv2d(nc, ndf, 1, 1, 0)

        layers = nn.ModuleList()

        ndf_cur = ndf
        for ind in range(self.depth):
            in_filters = ndf_cur
            ndf_cur = ndf_cur * 2
            for _ in range(res_blocks[ind]):
                layers.append(ResnetBlock(
                    in_channels=in_filters, 
                    out_channels=ndf_cur
                ))
                in_filters = ndf_cur
            img_shape = (tuple(int(d/2) for d in img_shape))
            layers.append(
                downsample_block_type(in_filters, ndf_cur))

        self.output_shape = (ndf_cur, *img_shape)
        self.layers = layers

    def forward(self, x):
        x = self.input_conv(x)
        for layer in self.layers:
            x = layer(x)
        return x

    def loss(self, x, y, layer_inds=None):
        if not layer_inds:
            layer_inds = [i for i in range(self.depth)]
        for layer_ind in layer_inds:
            assert layer_ind < self.depth, \
                f'layer={layer_ind} > depth={self.depth}'
        layer_inds = set(layer_inds)

        batch_size = x.shape[0]
        x = self.input_conv(x)
        y = self.input_conv(y)
        sum = 0

        for ind, layer in enumerate(self.layers):
            x = layer(x)
            y = layer(y)
            if ind in layer_inds:
                rx = x.reshape(batch_size, -1)
                ry = y.reshape(batch_size, -1)
                sum = sum + ((rx - ry)**2).mean(-1)
        return sum