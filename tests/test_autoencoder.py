import pytest
from model.autoencoders import AutoEncoder, VarAutoEncoder
import torch


@pytest.mark.parametrize("res_blocks", [(0, 0, 0), (1, 1, 1), (1, 2, 0)])
def test_auto_encoder(res_blocks):
    autoencoder = AutoEncoder(
        3, 16,
        latent_dim=516,
        depth=3,
        img_shape=(32, 32),
        res_blocks=res_blocks,
    )

    t_shape = (64, 3, 32, 32)
    t = torch.zeros(t_shape)
    y, *_ = autoencoder(t)
    y.shape == t_shape


@pytest.mark.parametrize("res_blocks", [(0, 0, 0), (1, 1, 1), (1, 2, 0)])
def test_var_auto_encoder(res_blocks):
    autoencoder = VarAutoEncoder(
        3, 16,
        latent_dim=516,
        depth=3,
        img_shape=(32, 32),
        res_blocks=res_blocks,
    )

    t_shape = (64, 3, 32, 32)
    t = torch.zeros(t_shape)
    y, mu, logvar = autoencoder(t)
    assert y.shape == t_shape
    assert mu.shape == (64, 516)
    assert logvar.shape == (64, 516)
