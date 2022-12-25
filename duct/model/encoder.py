import torch.nn as nn
from duct.model.resnet import ResnetBlock
from duct.model.torch_modules import get_conv, get_norm
from duct.model.attention import get_attn

class DownSampleBlock(nn.Module):
    def __init__(
            self, 
            in_filters, 
            out_filters, 
            norm_type='batch', 
            data_dim=2, 
            kernel=24, 
            stride=4, 
            padding=12):
        super().__init__()
        self.conv = get_conv(data_dim)(in_filters, out_filters, kernel, stride, padding)
        self.norm = get_norm(type=norm_type, data_dim=data_dim)(out_filters)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.leakyrelu(x)
        return x

    @classmethod
    def audio_block(cls, in_filters, out_filters):
        return cls(in_filters, out_filters, norm_type='batch', 
                   data_dim=1, kernel=24, stride=4, padding=11)

    @classmethod
    def audio_block_v2(cls, in_filters, out_filters):
        return cls(in_filters, out_filters, norm_type='batch', 
                   data_dim=1, kernel=12, stride=2, padding=5)

    @classmethod
    def image_block(cls, in_filters, out_filters):
        return cls(in_filters, out_filters, norm_type='batch',
                   data_dim=2, kernel=4, stride=2, padding=1)


class Encoder(nn.Module):
    def __init__(
            self, nc, ndf, data_shape,
            depth=5, 
            res_blocks=tuple(0 for _ in range(5)),
            attn_blocks=tuple(0 for _ in range(5)),
            downsample_block_type='image_block',
        ):
        super(Encoder, self).__init__()

        assert len(res_blocks) == depth, 'len(res_blocks) != depth'
        assert len(attn_blocks) == depth, 'len(attn_blocks) != depth'
        self.data_dim = len(data_shape)
        self.nc = nc
        self.ndf = ndf
        self.depth = depth
        self.input_conv = get_conv(self.data_dim)(nc, ndf, 1, 1, 0)

        layers = nn.ModuleList()

        ndf_cur = ndf
        for ind in range(self.depth):
            in_filters = ndf_cur
            ndf_cur = ndf_cur * 2
            for _ in range(res_blocks[ind]):
                layers.append(ResnetBlock(
                    in_channels=in_filters, 
                    out_channels=ndf_cur,
                    data_dim=self.data_dim,
                ))
                in_filters = ndf_cur
            for _ in range(attn_blocks[ind]):
                layers.append(get_attn(data_dim=self.data_dim)(in_filters))
            factor = 2 if downsample_block_type == 'image_block' else 4
            data_shape = (tuple(int(d/factor) for d in data_shape))
            layers.append(
                getattr(DownSampleBlock, downsample_block_type)(
                    in_filters, ndf_cur, 
                ))

        self.output_shape = (ndf_cur, *data_shape)
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