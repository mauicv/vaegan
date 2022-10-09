import torch.nn as nn
import numpy as np


class DownSampleBlock(nn.Module):
  def __init__(self, in_filters, out_filters, norm_type='instance'):
    super().__init__()
    self.conv = nn.Conv2d(in_filters, out_filters, 4, 2, 1)
    if norm_type == 'instance':
        self.norm = nn.InstanceNorm2d(out_filters)
    else:
        self.norm = nn.BatchNorm2d(out_filters)
    self.leakyrelu = nn.LeakyReLU(0.2)

  def forward(self, x):
    return self.leakyrelu(self.norm(self.conv(x)))


class UpSampleBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super().__init__()
        self.conv = nn.Conv2d(in_filters, out_filters, 3, 1)
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        self.rp2d = nn.ReplicationPad2d(1)
        self.bn = nn.BatchNorm2d(out_filters, 1.e-3)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.leakyrelu(self.bn(self.conv(self.rp2d(self.up(x)))))


class Encoder(nn.Module):
    def __init__(
            self, nc, ndf, depth=5,
            img_shape=(32, 32),
            norm_type='instance',
        ):
        super(Encoder, self).__init__()
        self.nc = nc
        self.ndf = ndf
        self.depth = depth
        self.input_conv = nn.Conv2d(nc, ndf, 1, 1, 0)

        layers = nn.ModuleList()

        ndf_cur = ndf
        for ind in range(1, self.depth):
            in_filters = ndf_cur
            ndf_cur = ndf_cur * 2
            img_shape = (tuple(int(d/2) for d in img_shape))
            layers.append(
                DownSampleBlock(in_filters, ndf_cur, norm_type=norm_type))

        self.encoding_output_shape = ndf_cur * np.prod(img_shape)
        self.layers = layers

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.input_conv(x)
        for layer in self.layers:
            x = layer(x)
        return x.view(batch_size, self.encoding_output_shape)

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
                sum = sum + ((rx - ry)**2).sum(-1)
        return sum


class Decoder(nn.Module):
    def __init__(
          self,
          nc, 
          ndf, 
          latent_variable_size, 
          depth=5,
          img_shape=(64, 64)):
        super(Decoder, self).__init__()

        self.nc = nc
        self.ndf = ndf
        self.depth = depth
        self.latent_variable_size = latent_variable_size
        self.out_conv = nn.Conv2d(ndf, nc, 1, 1)

        layers = nn.ModuleList()

        ndf_cur = ndf
        for ind in range(self.depth):
            img_shape = tuple(int(d/2) for d in img_shape)
            out_filters = ndf_cur
            ndf_cur = ndf_cur * 2
            layers.append(UpSampleBlock(ndf_cur, out_filters))

        self.layers = layers[::-1]

        self.encoding_shape = img_shape
        self.encoding_filters = ndf_cur
        self.d1 = nn.Linear(latent_variable_size, ndf_cur * np.prod(img_shape))

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
