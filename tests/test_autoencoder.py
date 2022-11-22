from model.autoencoders import AutoEncoder
import torch


def test_auto_encoder():
    autoencoder = AutoEncoder(
        3, 16,
        latent_dim=10,
        depth=3,
        img_shape=(32, 32))

    t_shape = (64, 3, 32, 32)
    t = torch.zeros(t_shape)
    autoencoder(t).shape == t_shape