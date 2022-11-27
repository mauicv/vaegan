import torch
from duct.model.latent_spaces import LinearLatentSpace, StochasticLinearLatentSpace, \
    StochasticLatentSpace


def test_linear_latent_space():
    latent_space = LinearLatentSpace((64, 8, 8), (64, 8, 8), 5)
    in_tensor = torch.randn((64, 64, 8, 8))
    out_tensor, *_ = latent_space(in_tensor)
    assert in_tensor.shape == out_tensor.shape

def test_stochastic_linear_latent_space():
    latent_space = StochasticLinearLatentSpace((64, 8, 8), (64, 8, 8), 5)
    in_tensor = torch.randn((64, 64, 8, 8))
    out_tensor, mu, logvar = latent_space(in_tensor)
    assert in_tensor.shape == out_tensor.shape
    assert mu.shape == (64, 5)
    assert logvar.shape == (64, 5)

def test_stochastic_latent_space():
    latent_space = StochasticLatentSpace((64, 8, 8), (64, 8, 8))
    in_tensor = torch.randn((64, 64, 8, 8))
    out_tensor, mu, logvar = latent_space(in_tensor)
    assert in_tensor.shape == out_tensor.shape
    assert mu.shape == out_tensor.shape
    assert logvar.shape == out_tensor.shape