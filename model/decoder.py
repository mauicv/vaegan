import torch.nn as nn
import numpy as np
from model.resnet import ResnetBlock

class UpSampleBatchConvBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super().__init__()
        self.conv = nn.Conv2d(in_filters, out_filters, 3, 1)
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        self.rp2d = nn.ReplicationPad2d(1)
        self.bn = nn.BatchNorm2d(out_filters, 1.e-3)
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
        ):
        super(Decoder, self).__init__()

        assert len(res_blocks) == depth

        self.nc = nc
        self.ndf = ndf
        self.depth = depth
        self.out_conv = nn.Conv2d(ndf, nc, 1, 1)

        layers = nn.ModuleList()

        ndf_cur = ndf
        for ind in range(self.depth):
            img_shape = tuple(int(d/2) for d in img_shape)
            out_filters = ndf_cur
            ndf_cur = ndf_cur * 2
            for _ in range(res_blocks[ind]):
                layers.append(ResnetBlock(ndf_cur, out_filters))
                out_filters = ndf_cur
            layers.append(upsample_block_type(ndf_cur, out_filters))

        self.layers = layers[::-1]
        self.input_shape = (ndf_cur, *img_shape)
        self.sig = nn.Sigmoid()

    def forward(self, z):
        for layer in self.layers:
            z = layer(z)
        return self.sig(self.out_conv(z))
