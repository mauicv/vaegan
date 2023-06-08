from duct.model.spectral_layers import SpectralTransform, SpectralResidualBlock, \
    SpectralEncoderBlock, SpectralEncoder
import torch


def test_spectral_transform():
    st = SpectralTransform(
        n_fft=1024, 
        hop_length=256, 
        window_length=1024
    )
    t = torch.randn((64, 1, 8192))
    assert st(t).shape == (64, 2, 513, 33)


def test_spectral_residual_block():
    srb = SpectralResidualBlock(
        in_channels=2,
        mid_channels=16,
        out_channels=32,
        s_t=2, s_f=2
    )
    t = torch.randn((64, 2, 513, 33))
    assert srb(t).shape == (64, 32, 257, 17)

def test_spectral_encoder_block():
    seb = SpectralEncoderBlock(
        in_channels=2,
        mid_channels=16,
        out_channels=32,
    )
    t = torch.randn((64, 2, 513, 33))
    assert seb(t).shape == (64, 32, 129, 17)


def test_spectral_encoder():
    st = SpectralTransform(
        n_fft=1024, 
        hop_length=256, 
        window_length=1024
    )
    s_enc = SpectralEncoder(
        nc=2,
        depth=3,
        ndf=8
    )

    t = torch.randn((64, 1, 8192))
    x = s_enc(st(t))
    assert x.shape == (64, 512, 8, 4)