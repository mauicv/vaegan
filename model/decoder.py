import torch.nn as nn
import numpy as np


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
          latent_dim, 
          depth=5,
          img_shape=(64, 64),
          upsample_block_type=UpSampleBatchConvBlock):
        super(Decoder, self).__init__()

        self.nc = nc
        self.ndf = ndf
        self.depth = depth
        self.latent_dim = latent_dim
        self.out_conv = nn.Conv2d(ndf, nc, 1, 1)

        layers = nn.ModuleList()

        ndf_cur = ndf
        for ind in range(self.depth):
            img_shape = tuple(int(d/2) for d in img_shape)
            out_filters = ndf_cur
            ndf_cur = ndf_cur * 2
            layers.append(upsample_block_type(ndf_cur, out_filters))

        self.layers = layers[::-1]

        self.encoding_shape = img_shape
        self.encoding_filters = ndf_cur
        self.d1 = nn.Linear(latent_dim, ndf_cur * np.prod(img_shape))

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, z):
        batch_size = z.shape[0]
        x = self.relu(self.d1(z))
        x = x.reshape(batch_size, self.encoding_filters, *self.encoding_shape)
        for layer in self.layers:
          x = layer(x)
        return self.sig(self.out_conv(x))
