import torch.nn as nn
import torch
from duct.model.decoder import Decoder
from duct.model.encoder import Encoder
from itertools import chain
from duct.model.torch_modules import get_nonlinearity


class BaseAutoEncoder(nn.Module):
    def __init__(
            self, 
            nc, 
            ndf, 
            data_shape,
            depth=5, 
            res_blocks=tuple(0 for _ in range(5)),
            attn_blocks=tuple(0 for _ in range(5)),
            ch_mult=(1, 1, 2, 2, 4),
            output_activation='sigmoid',
            downsample_block_type='image_block',
            upsample_block_type='image_block'
        ):
        super(BaseAutoEncoder, self).__init__()
        assert len(data_shape) in {1, 2}, "data_dim must be 1 or 2"
        assert len(res_blocks) == len(attn_blocks) == len(ch_mult),(
                f'len(res_blocks)={len(res_blocks)},'
                f'len(attn_blocks)={len(attn_blocks)} and '
                f'len(ch_mult)={len(ch_mult)} should all be the same length'
            )
        self.encoder = Encoder(
            nc=nc, ndf=ndf, depth=depth, 
            data_shape=data_shape,
            res_blocks=res_blocks,
            ch_mult=ch_mult,
            downsample_block_type=downsample_block_type,
            attn_blocks=attn_blocks
        )
        self.decoder = Decoder(
            nc=nc, ndf=ndf, depth=depth,
            data_shape=data_shape,
            res_blocks=res_blocks,
            ch_mult=ch_mult,
            upsample_block_type=upsample_block_type,
            attn_blocks=attn_blocks
        )
        self.output_activation = get_nonlinearity(output_activation) \
            if output_activation else None

    def forward(self, x):
        x = self.encoder(x)
        out_z = self.latent_space(x)
        y = self.decoder(out_z[0])
        if self.output_activation is not None:
            y = self.output_activation(y)
        return (y, *out_z[1:])

    def encode(self, x):
        return self.latent_space(self.encoder(x))

    def decode(self, z):
        return self.decoder(self.latent_space.decode(z))

    def encoder_params(self):
        return chain(
            self.encoder.parameters(), 
            self.latent_space.parameters(), 
        )

    def decoder_params(self):
        return self.decoder.parameters()