import pytest
from duct.model.autoencoders import AutoEncoder2D, VarAutoEncoder2D, NLLVarAutoEncoder2D, \
    VQVarAutoEncoder2D, VQVarAutoEncoder1D
import torch


@pytest.mark.parametrize("res_blocks", [(0, 0, 0), (1, 1, 1), (1, 2, 0)])
def test_auto_encoder(res_blocks):
    autoencoder = AutoEncoder2D(
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
    autoencoder = VarAutoEncoder2D(
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
    assert autoencoder.call(t).shape == t_shape


@pytest.mark.parametrize("res_blocks", [(0, 0, 0), (1, 1, 1), (1, 2, 0)])
def test_nll_var_auto_encoder(res_blocks):
    autoencoder = NLLVarAutoEncoder2D(
        3, 16,
        latent_dim=None,
        depth=3,
        img_shape=(32, 32),
        res_blocks=res_blocks,
    )

    t_shape = (64, 3, 32, 32)
    t = torch.zeros(t_shape)
    y, mu, logvar = autoencoder(t)
    assert y.shape == t_shape
    assert mu.shape == (64, 128, 4, 4)
    assert logvar.shape == (64, 128, 4, 4)
    assert autoencoder.call(t).shape == t_shape


@pytest.mark.parametrize("res_blocks", [(0, 0, 0), (1, 1, 1), (1, 2, 0)])
def test_vq_var_auto_encoder_2d(res_blocks):
    autoencoder = VQVarAutoEncoder2D(
        3, 16,
        latent_dim=None,
        depth=3,
        img_shape=(32, 32),
        res_blocks=res_blocks,
        commitment_cost=1,
        num_embeddings=100
    )

    t_shape = (64, 3, 32, 32)
    t = torch.zeros(t_shape)
    y, loss, perplexity, encoded = autoencoder(t)
    assert y.shape == t_shape
    assert encoded.shape == (64, 4, 4, 100)
    assert autoencoder.call(t).shape == t_shape


# @pytest.mark.xfail(reason="Not implemented")
@pytest.mark.parametrize("res_blocks", [(0, 0, 0), (1, 1, 1), (1, 2, 0)])
def test_vq_var_auto_encoder_1d(res_blocks):
    autoencoder = VQVarAutoEncoder1D(
        2, 16,
        latent_dim=None,
        depth=3,
        img_shape=(32, 32),
        res_blocks=res_blocks,
        commitment_cost=1,
        num_embeddings=100
    )

    t_shape = (64, 2, 32)
    t = torch.zeros(t_shape)
    y, loss, perplexity, encoded = autoencoder(t)
    print(encoded.shape)
    print(y.shape)
    assert y.shape == t_shape
    # assert encoded.shape == (64, 4, 4, 100)
    # assert autoencoder.call(t).shape == t_shape
