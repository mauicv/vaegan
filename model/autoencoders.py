from model.latent_spaces import LatentSpace, StochasticLatentSpace
from model.base_autoencoder import BaseAutoEncoder
import torch.nn as nn


class AutoEncoder(BaseAutoEncoder):
    def __init__(
            self, 
            nc, 
            ndf, 
            latent_dim,
            depth=5, 
            img_shape=(64, 64),
        ):
        super(AutoEncoder, self).__init__(
            nc=nc, ndf=ndf, depth=depth, 
            img_shape=img_shape
        )
        self.latent_space = LatentSpace(
            input_shape=self.encoder.output_shape, 
            output_shape=self.decoder.input_shape, 
            latent_dim=latent_dim
        )


class VarAutoEncoder(BaseAutoEncoder):
    def __init__(
            self, 
            nc, 
            ndf, 
            latent_dim,
            depth=5, 
            img_shape=(64, 64),
        ):
        super(VarAutoEncoder, self).__init__(
            nc=nc, ndf=ndf, depth=depth, 
            img_shape=img_shape
        )
        self.latent_space = StochasticLatentSpace(
            input_shape=self.encoder.output_shape, 
            output_shape=self.decoder.input_shape, 
            latent_dim=latent_dim
        )
