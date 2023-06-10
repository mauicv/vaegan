from duct.utils.losses import SpectralLoss, MultiSpectralLoss
import torch


def test_spectral_loss():
    spec_loss = SpectralLoss(window_length=2**11, device="cpu")
    t1 = torch.randn((2, 1, 24000))
    t2 = torch.randn((2, 1, 24000))
    sl = spec_loss(t1, t2, reduction=None)
    assert sl.shape == (2, )

    sl = spec_loss(t1, t2)
    assert sl > 0

    t = torch.randn((1, 1, 24000))
    sl = spec_loss(t, t)
    assert sl == 0


def test_multi_spectral_loss():
    spec_loss = MultiSpectralLoss(device="cpu")
    t1 = torch.randn((2, 1, 24000))
    t2 = torch.randn((2, 1, 24000))
    sl = spec_loss(t1, t2)
    assert sl > 0

    t = torch.randn((1, 1, 24000))
    sl = spec_loss(t, t)
    assert sl == 0