import pytest
from duct.model.autoencoders import AutoEncoder, VarAutoEncoder, NLLVarAutoEncoder, \
    VQVarAutoEncoder
import torch


@pytest.mark.parametrize("res_blocks", [(0, 0, 0), (1, 1, 1), (1, 2, 0)])
def test_auto_encoder(res_blocks):
    autoencoder = AutoEncoder(
        3, 16,
        latent_dim=516,
        depth=3,
        data_shape=(32, 32),
        res_blocks=res_blocks,
        attn_blocks=(0, 0, 0),
    )

    t_shape = (64, 3, 32, 32)
    t = torch.randn(t_shape)
    y, *_ = autoencoder(t)
    y.shape == t_shape


@pytest.mark.parametrize("res_blocks", [(0, 0, 0), (1, 1, 1), (1, 2, 0)])
def test_var_auto_encoder(res_blocks):
    autoencoder = VarAutoEncoder(
        3, 16,
        latent_dim=516,
        depth=3,
        data_shape=(32, 32),
        res_blocks=res_blocks,
        attn_blocks=(0, 0, 0),
    )

    t_shape = (64, 3, 32, 32)
    t = torch.randn(t_shape)
    y, mu, logvar = autoencoder(t)
    assert y.shape == t_shape
    assert mu.shape == (64, 516)
    assert logvar.shape == (64, 516)
    assert autoencoder.call(t).shape == t_shape


@pytest.mark.parametrize("res_blocks", [(0, 0, 0), (1, 1, 1), (1, 2, 0)])
def test_nll_var_auto_encoder(res_blocks):
    autoencoder = NLLVarAutoEncoder(
        3, 16,
        latent_dim=None,
        depth=3,
        data_shape=(32, 32),
        res_blocks=res_blocks,
        attn_blocks=(0, 0, 1),
    )

    t_shape = (64, 3, 32, 32)
    t = torch.randn(t_shape)
    y, mu, logvar = autoencoder(t)
    assert y.shape == t_shape
    assert mu.shape == (64, 128, 4, 4)
    assert logvar.shape == (64, 128, 4, 4)
    assert autoencoder.call(t).shape == t_shape


@pytest.mark.parametrize("res_blocks", [(0, 0, 0), (1, 1, 1), (1, 2, 0)])
def test_vq_var_auto_encoder_2d(res_blocks):
    autoencoder = VQVarAutoEncoder(
        3, 16,
        latent_dim=None,
        depth=3,
        data_shape=(32, 32),
        res_blocks=res_blocks,
        attn_blocks=(0, 0, 1),
        commitment_cost=1,
        num_embeddings=100,
    )

    t_shape = (64, 3, 32, 32)
    t = torch.randn(t_shape)
    y, _, _, encoded = autoencoder(t)
    assert y.shape == t_shape
    assert encoded.shape == (64, 4, 4, 100)
    assert autoencoder.call(t).shape == t_shape


@pytest.mark.parametrize("res_blocks", [(0, 0, 0, 0), (1, 1, 1, 1), (1, 2, 0, 0)])
def test_vq_var_auto_encoder_1d(res_blocks):
    autoencoder = VQVarAutoEncoder(
        2, 16,
        latent_dim=None,
        depth=4,
        data_shape=(8192, ),
        res_blocks=res_blocks,
        attn_blocks=(0, 0, 0, 1),
        commitment_cost=1,
        num_embeddings=100,
        output_activation='sigmoid',
        upsample_block_type='audio_block',
        downsample_block_type='audio_block'
    )

    t_shape = (64, 2, 8192)
    t = torch.randn(t_shape)
    y, _, _, encoded = autoencoder(t)
    assert y.shape == t_shape
    assert encoded.shape == (64, 32, 100)
    assert autoencoder.call(t).shape == t_shape


def test_vq_var_auto_encoder_1d_aud():
    autoencoder = VQVarAutoEncoder(
        2, 16, depth=5,
        data_shape=(8192, ),
        res_blocks=(0, 0, 0, 0, 0),
        attn_blocks=(0, 0, 0, 0, 0),
        commitment_cost=1,
        num_embeddings=100,
        output_activation='tanh',
        upsample_block_type='audio_block',
        downsample_block_type='audio_block'
    )

    t_shape = (64, 2, 8192)
    t = torch.randn(t_shape)
    y, _, _, encoded = autoencoder(t)
    assert -1 <= y.min() < y.max() <= 1
    assert y.shape == t_shape
    assert encoded.shape == (64, 8, 100)
    assert autoencoder.call(t).shape == t_shape


def test_vq_var_auto_encoder_1d_v2_aud():
    autoencoder = VQVarAutoEncoder(
        2, 16, depth=3,
        data_shape=(8192, ),
        res_blocks=(0, 0, 0),
        attn_blocks=(0, 0, 0),
        commitment_cost=1,
        num_embeddings=100,
        output_activation='tanh',
        upsample_block_type='audio_block_v2',
        downsample_block_type='audio_block_v2'
    )

    t_shape = (64, 2, 8192)
    t = torch.randn(t_shape)
    y, _, _, encoded = autoencoder(t)
    assert -1 <= y.min() < y.max() <= 1
    assert encoded.shape == (64, 1024, 100)
    assert y.shape == t_shape
    assert autoencoder.call(t).shape == t_shape
    assert autoencoder.encode(t).shape == encoded.shape