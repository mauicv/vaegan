from duct.model.latent_spaces.continuous import LinearLatentSpace, StochasticLinearLatentSpace, \
    StochasticLatentSpace
from duct.model.latent_spaces.discrete import VQLatentSpace2D, VQLatentSpace1D
from duct.model.base_autoencoder import BaseAutoEncoder


class AutoEncoder(BaseAutoEncoder):
    def __init__(
            self, 
            nc, 
            ndf, 
            data_shape,
            depth=5, 
            res_blocks=tuple(0 for _ in range(5)),
            attn_blocks=tuple(0 for _ in range(5)),
            ch_mult=(1, 1, 2, 2, 4),
            latent_dim=None,
            downsample_block_type='image_block',
            upsample_block_type='image_block',
        ):
        super(AutoEncoder, self).__init__(
            nc=nc, ndf=ndf, depth=depth, 
            data_shape=data_shape,
            res_blocks=res_blocks,
            ch_mult=ch_mult,
            downsample_block_type=downsample_block_type,
            upsample_block_type=upsample_block_type,
            attn_blocks=attn_blocks
        )
        self.latent_space = LinearLatentSpace(
            input_shape=self.encoder.output_shape, 
            output_shape=self.decoder.input_shape, 
            latent_dim=latent_dim
        )


class VarAutoEncoder(BaseAutoEncoder):
    def __init__(
            self, 
            nc, 
            ndf, 
            data_shape,
            depth=5, 
            res_blocks=tuple(0 for _ in range(5)),
            attn_blocks=tuple(0 for _ in range(5)),
            ch_mult=(1, 1, 2, 2, 4),
            latent_dim=None,
            downsample_block_type='image_block',
            upsample_block_type='image_block'
        ):
        super(VarAutoEncoder, self).__init__(
            nc=nc, ndf=ndf, depth=depth, 
            data_shape=data_shape,
            res_blocks=res_blocks,
            ch_mult=ch_mult,
            downsample_block_type=downsample_block_type,
            upsample_block_type=upsample_block_type,
            attn_blocks=attn_blocks
        )
        self.latent_space = StochasticLinearLatentSpace(
            input_shape=self.encoder.output_shape, 
            output_shape=self.decoder.input_shape, 
            latent_dim=latent_dim
        )

    def call(self, x):
        x = self.encoder(x)
        out_z = self.latent_space(x)
        return self.decoder(out_z[0])


class NLLVarAutoEncoder(BaseAutoEncoder):
    def __init__(
            self, 
            nc, 
            ndf, 
            data_shape,
            depth=5, 
            res_blocks=tuple(0 for _ in range(5)),
            attn_blocks=tuple(0 for _ in range(5)),
            ch_mult=(1, 1, 2, 2, 4),
            latent_dim=None,
            downsample_block_type='image_block',
            upsample_block_type='image_block'
        ):
        assert latent_dim is None

        super(NLLVarAutoEncoder, self).__init__(
            nc=nc, ndf=ndf, depth=depth, 
            data_shape=data_shape,
            res_blocks=res_blocks,
            ch_mult=ch_mult,
            downsample_block_type=downsample_block_type,
            upsample_block_type=upsample_block_type,
            attn_blocks=attn_blocks
        )
        self.latent_space = StochasticLatentSpace(
            input_shape=self.encoder.output_shape, 
            output_shape=self.decoder.input_shape, 
            latent_dim=latent_dim,
        )

    def call(self, x):
        x = self.encoder(x)
        out_z = self.latent_space(x)
        return self.decoder(out_z[0])


class VQVarAutoEncoder(BaseAutoEncoder):
    def __init__(
            self, 
            nc, 
            ndf, 
            data_shape,
            depth=5, 
            res_blocks=tuple(0 for _ in range(5)),
            attn_blocks=tuple(0 for _ in range(5)),
            ch_mult=(1, 1, 2, 2, 4),
            num_embeddings=25,
            commitment_cost=1,
            output_activation='sigmoid',
            latent_dim=None,
            downsample_block_type='image_block',
            upsample_block_type='image_block'
        ):
        assert latent_dim is None
        data_dim=len(data_shape)

        super(VQVarAutoEncoder, self).__init__(
            nc=nc, ndf=ndf, depth=depth, 
            data_shape=data_shape,
            res_blocks=res_blocks,
            ch_mult=ch_mult,
            output_activation=output_activation,
            downsample_block_type=downsample_block_type,
            upsample_block_type=upsample_block_type,
            attn_blocks=attn_blocks
        )

        C, *_ = self.encoder.output_shape

        self.latent_space = {
            1: VQLatentSpace1D,
            2: VQLatentSpace2D
        }[data_dim](
            num_embeddings=num_embeddings, 
            embedding_dim=C, 
            commitment_cost=commitment_cost
        )

    def call(self, x, training=True):
        x = self.encoder(x)
        out_z = self.latent_space(x, training=training)
        y = self.decoder(out_z[0])
        if self.output_activation is not None:
            y = self.output_activation(y)
        return y

    def encode(self, x):
        x = self.encoder(x)
        out_z = self.latent_space(x, training=False)
        return out_z[-1]
