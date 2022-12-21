import torch.nn as nn
from duct.model.decoder import Decoder, UpSampleBatchConvBlock
from duct.model.encoder import Encoder, DownSampleBatchConvBlock
from itertools import chain


class BaseAutoEncoder(nn.Module):
    def __init__(
            self, 
            nc, 
            ndf, 
            depth=5, 
            img_shape=(64, 64),
            res_blocks=tuple(0 for _ in range(5)),
            data_dim=2
        ):
        super(BaseAutoEncoder, self).__init__()
        self.encoder = Encoder(
            nc=nc, ndf=ndf, depth=depth, 
            img_shape=img_shape,
            res_blocks=res_blocks,
            downsample_block_type=DownSampleBatchConvBlock,
            data_dim=data_dim
        )
        self.decoder = Decoder(
            nc=nc, ndf=ndf, depth=depth,
            img_shape=img_shape,
            res_blocks=res_blocks,
            upsample_block_type=UpSampleBatchConvBlock,
            data_dim=data_dim
        )

    def forward(self, x):
        x = self.encoder(x)
        out_z = self.latent_space(x)
        return (self.decoder(out_z[0]), *out_z[1:])

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