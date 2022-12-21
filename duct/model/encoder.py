import torch.nn as nn
import numpy as np
from duct.model.resnet import ResnetBlock
from duct.model.torch_modules import get_conv, get_instance_norm, get_batch_norm


class DownSampleInstanceConvBlock(nn.Module):
    def __init__(self, in_filters, out_filters, data_dim=2):
        super().__init__()
        self.conv = get_conv(data_dim)(in_filters, out_filters, 4, 2, 1)
        self.norm = get_instance_norm(data_dim)(out_filters)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.leakyrelu(self.norm(self.conv(x)))


class DownSampleBatchConvBlock(nn.Module):
    def __init__(self, in_filters, out_filters, data_dim=2):
        super().__init__()
        self.conv = get_conv(data_dim)(in_filters, out_filters, 4, 2, 1)
        self.norm = get_batch_norm(data_dim)(out_filters)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.leakyrelu(self.norm(self.conv(x)))


class Encoder(nn.Module):
    def __init__(
            self, nc, ndf, depth=5,
            img_shape=(32, 32),
            res_blocks=tuple(0 for _ in range(5)),
            downsample_block_type=DownSampleInstanceConvBlock,
            data_dim=2
        ):
        super(Encoder, self).__init__()

        assert len(res_blocks) == depth, 'len(res_blocks) != depth'

        self.nc = nc
        self.ndf = ndf
        self.depth = depth
        self.input_conv = get_conv(data_dim)(nc, ndf, 1, 1, 0)

        layers = nn.ModuleList()

        ndf_cur = ndf
        for ind in range(self.depth):
            in_filters = ndf_cur
            ndf_cur = ndf_cur * 2
            for _ in range(res_blocks[ind]):
                layers.append(ResnetBlock(
                    in_channels=in_filters, 
                    out_channels=ndf_cur,
                    data_dim=data_dim,
                ))
                in_filters = ndf_cur
            img_shape = (tuple(int(d/2) for d in img_shape))
            layers.append(
                downsample_block_type(
                    in_filters, ndf_cur, 
                    data_dim=data_dim
                ))

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