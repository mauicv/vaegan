from model.autoencoders import AutoEncoder, VarAutoEncoder
import torch


def test_auto_encoder():
    autoencoder = AutoEncoder(
        3, 16,
        latent_dim=516,
        depth=3,
        img_shape=(32, 32)
    )

    t_shape = (64, 3, 32, 32)
    t = torch.zeros(t_shape)
    y, *_ = autoencoder(t)
    y.shape == t_shape


def test_var_auto_encoder():
    autoencoder = VarAutoEncoder(
        3, 16,
        latent_dim=516,
        depth=3,
        img_shape=(32, 32)
    )

    t_shape = (64, 3, 32, 32)
    t = torch.zeros(t_shape)
    y, mu, logvar = autoencoder(t)
    assert y.shape == t_shape
    assert mu.shape == (64, 516)
    assert logvar.shape == (64, 516)
