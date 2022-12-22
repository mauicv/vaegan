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
            latent_dim=None,
        ):
        super(AutoEncoder, self).__init__(
            nc=nc, ndf=ndf, depth=depth, 
            data_shape=data_shape,
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
            data_shape,
            depth=5, 
            res_blocks=tuple(0 for _ in range(5)),
            latent_dim=None,
        ):
        super(VarAutoEncoder, self).__init__(
            nc=nc, ndf=ndf, depth=depth, 
            data_shape=data_shape,
            res_blocks=res_blocks,
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
            latent_dim=None,
        ):
        assert latent_dim is None

        super(NLLVarAutoEncoder, self).__init__(
            nc=nc, ndf=ndf, depth=depth, 
            data_shape=data_shape,
            res_blocks=res_blocks,
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
            data_shape,
            depth=5, 
            res_blocks=tuple(0 for _ in range(5)),
            num_embeddings=25,
            commitment_cost=1,
            output_activation='Sigmoid',
            latent_dim=None,
        ):
        assert latent_dim is None
        data_dim=len(data_shape)

        super(VQVarAutoEncoder, self).__init__(
            nc=nc, ndf=ndf, depth=depth, 
            data_shape=data_shape,
            res_blocks=res_blocks,
            output_activation=output_activation
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

    def call(self, x):
        x = self.encoder(x)
        out_z = self.latent_space(x)
        return self.decoder(out_z[0])
