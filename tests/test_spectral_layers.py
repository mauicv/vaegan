from duct.model.spectral_layers import SpectralTransform, SpectralEncoder
import torch


def test_spectral_transform():
    st = SpectralTransform(
        n_fft=1024, 
        hop_length=256, 
        window_length=1024
    )
    t = torch.randn((64, 1, 8192))
    assert st(t).shape == (64, 2, 29, 513)


def test_spectral_encoder():
    st = SpectralTransform(
        n_fft=1024, 
        hop_length=256, 
        window_length=1024
    )
    s_enc = SpectralEncoder(nc=1)

    t = torch.randn((64, 1, 8192))
    x, _ = s_enc(st(t))
    assert x.shape == (64, 256, 27, 63)