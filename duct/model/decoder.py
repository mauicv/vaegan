import torch.nn as nn
from duct.model.resnet import ResnetBlock
from duct.model.torch_modules import get_conv, get_norm, get_rep_pad, get_upsample
from duct.model.attention import get_attn
from duct.model.activations import get_nonlinearity


class UpSampleBlock(nn.Module):
    def __init__(
            self, 
            in_filters, 
            out_filters, 
            data_dim, 
            norm_type, 
            scale_factor, 
            kernel, 
            padding,
            dropout=0.1,
            activation='leakyrelu'):
        super().__init__()
        self.conv = get_conv(data_dim)(in_filters, out_filters, kernel, padding=0)
        self.up = get_upsample(data_dim)(scale_factor=scale_factor)
        self.rp2d = get_rep_pad(data_dim)(padding)
        self.norm = get_norm(type=norm_type, data_dim=data_dim)(out_filters, 1.e-3)
        self.nonlinearity = get_nonlinearity(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.up(x)
        x = self.rp2d(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.nonlinearity(x)
        x = self.dropout(x)
        return x

    @classmethod
    def audio_block(cls, in_filters, out_filters):
        return cls(in_filters, out_filters, norm_type='batch', data_dim=1, 
                   scale_factor=4, kernel=25, padding=12, dropout=0.1)

    @classmethod
    def audio_block_v2(cls, in_filters, out_filters):
        return cls(in_filters, out_filters, norm_type='batch', data_dim=1,
                   scale_factor=2, kernel=11, padding=5, dropout=0.1)

    @classmethod
    def image_block(cls, in_filters, out_filters):
        return cls(in_filters, out_filters, norm_type='batch', data_dim=2, 
                   scale_factor=2, kernel=3, padding=1, dropout=0.1)


class Decoder(nn.Module):
    def __init__(
            self,
            nc, 
            ndf,
            data_shape,
            depth=5,
            res_blocks=tuple(0 for _ in range(5)),
            attn_blocks=tuple(0 for _ in range(5)),
            upsample_block_type='image_block',
        ):
        super(Decoder, self).__init__()

        assert len(res_blocks) == depth, 'len(res_blocks) != depth'
        assert len(attn_blocks) == depth, 'len(attn_blocks) != depth'

        self.data_dim = len(data_shape)

        self.nc = nc
        self.ndf = ndf
        self.depth = depth
        self.output_conv = get_conv(self.data_dim)(ndf, nc, 1, 1)
        self.output_norm = get_norm(type='batch', data_dim=self.data_dim)(nc, 1.e-3)

        layers = nn.ModuleList()

        ndf_cur = ndf
        for ind in range(self.depth):
            data_shape = tuple(int(d/2) for d in data_shape)
            out_filters = ndf_cur
            ndf_cur = ndf_cur * 2
            for _ in range(res_blocks[ind]):
                layers.append(ResnetBlock(ndf_cur, out_filters, 
                    data_dim=self.data_dim))
                out_filters = ndf_cur
            for _ in range(attn_blocks[ind]):
                layers.append(get_attn(self.data_dim)(out_filters))
            layers.append(getattr(UpSampleBlock, upsample_block_type)(ndf_cur, out_filters))

        self.layers = layers[::-1]
        self.input_shape = (ndf_cur, *data_shape)

    def forward(self, z):
        for layer in self.layers:
            z = layer(z)
        x = self.output_conv(z)
        x = self.output_norm(x)
        return x