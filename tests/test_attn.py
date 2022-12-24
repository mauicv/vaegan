import torch
from duct.model.attention import AttnBlock1D, AttnBlock2D


def test_attention_1d():
    attn_block = AttnBlock1D(in_channels=2)
    x = torch.randn(1, 2, 32)
    y = attn_block(x)
    assert y.shape == x.shape


def test_attention2d():
    attn_block = AttnBlock2D(in_channels=3)
    x = torch.randn(1, 3, 32, 32)
    y = attn_block(x)
    assert y.shape == x.shape
    