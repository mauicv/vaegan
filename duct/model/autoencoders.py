from duct.model.latent_spaces import LinearLatentSpace, StochasticLinearLatentSpace, \
    StochasticLatentSpace, VQLatentSpace
from duct.model.base_autoencoder import BaseAutoEncoder


class AutoEncoder(BaseAutoEncoder):
    def __init__(
            self, 
            nc, 
            ndf, 
            depth=5, 
            img_shape=(64, 64),
            res_blocks=tuple(0 for _ in range(5)),
            latent_dim=None,
        ):
        super(AutoEncoder, self).__init__(
            nc=nc, ndf=ndf, depth=depth, 
            img_shape=img_shape,
            res_blocks=res_blocks,
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
            depth=5, 
            img_shape=(64, 64),
            res_blocks=tuple(0 for _ in range(5)),
            latent_dim=None,
        ):
        super(VarAutoEncoder, self).__init__(
            nc=nc, ndf=ndf, depth=depth, 
            img_shape=img_shape,
            res_blocks=res_blocks
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
            depth=5, 
            img_shape=(64, 64),
            res_blocks=tuple(0 for _ in range(5)),
            latent_dim=None,
        ):
        assert latent_dim is None

        super(NLLVarAutoEncoder, self).__init__(
            nc=nc, ndf=ndf, depth=depth, 
            img_shape=img_shape,
            res_blocks=res_blocks
        )
        self.latent_space = StochasticLatentSpace(
            input_shape=self.encoder.output_shape, 
            output_shape=self.decoder.input_shape, 
            latent_dim=latent_dim
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
            depth=5, 
            img_shape=(64, 64),
            res_blocks=tuple(0 for _ in range(5)),
            num_embeddings=25,
            commitment_cost=1,
            latent_dim=None
        ):
        assert latent_dim is None

        super(VQVarAutoEncoder, self).__init__(
            nc=nc, ndf=ndf, depth=depth, 
            img_shape=img_shape,
            res_blocks=res_blocks
        )

        C, _, _ = self.encoder.output_shape

        self.latent_space = VQLatentSpace(
            num_embeddings=num_embeddings, 
            embedding_dim=C, 
            commitment_cost=commitment_cost
        )

    def call(self, x):
        x = self.encoder(x)
        out_z = self.latent_space(x)
        return self.decoder(out_z[0])
