import torch
from model.latent_spaces import LatentSpace, StochasticLatentSpace


def test_latent_space():
    latent_space = LatentSpace((64, 8, 8), (64, 8, 8), 5)
    in_tensor = torch.randn((64, 64, 8, 8))
    out_tensor = latent_space(in_tensor)
    assert in_tensor.shape == out_tensor.shape

def test_stochastic_latent_space():
    latent_space = StochasticLatentSpace((64, 8, 8), (64, 8, 8), 5)
    in_tensor = torch.randn((64, 64, 8, 8))
    out_tensor, mu, logvar = latent_space(in_tensor)
    assert in_tensor.shape == out_tensor.shape
    assert mu.shape == (64, 5)
    assert logvar.shape == (64, 5)