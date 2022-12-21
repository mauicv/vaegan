import torch.nn as nn
import numpy as np
from duct.model.resnet import ResnetBlock
from duct.model.torch_modules import get_conv, get_batch_norm, get_rep_pad, get_upsample

class UpSampleBatchConvBlock(nn.Module):
    def __init__(self, in_filters, out_filters, data_dim=2):
        super().__init__()
        self.conv = get_conv(data_dim)(in_filters, out_filters, 3, 1)
        self.up = get_upsample(data_dim)(scale_factor=2)
        self.rp2d = get_rep_pad(data_dim)(1)
        self.bn = get_batch_norm(data_dim=data_dim)(out_filters, 1.e-3)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.leakyrelu(self.bn(self.conv(self.rp2d(self.up(x)))))


class Decoder(nn.Module):
    def __init__(
            self,
            nc, 
            ndf,
            depth=5,
            img_shape=(64, 64),
            upsample_block_type=UpSampleBatchConvBlock,
            res_blocks=tuple(0 for _ in range(5)),
            data_dim=2
        ):
        super(Decoder, self).__init__()

        assert len(res_blocks) == depth

        self.nc = nc
        self.ndf = ndf
        self.depth = depth
        self.out_conv = get_conv(data_dim)(ndf, nc, 1, 1)

        layers = nn.ModuleList()

        ndf_cur = ndf
        for ind in range(self.depth):
            img_shape = tuple(int(d/2) for d in img_shape)
            out_filters = ndf_cur
            ndf_cur = ndf_cur * 2
            for _ in range(res_blocks[ind]):
                layers.append(ResnetBlock(ndf_cur, out_filters, 
                    data_dim=data_dim))
                out_filters = ndf_cur
            layers.append(upsample_block_type(ndf_cur, out_filters, 
                data_dim=data_dim))

        self.layers = layers[::-1]
        self.input_shape = (ndf_cur, *img_shape)
        self.sig = nn.Sigmoid()

    def forward(self, z):
        for layer in self.layers:
            z = layer(z)
        return self.sig(self.out_conv(z))
