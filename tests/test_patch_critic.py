import pytest
import torch
from duct.model.patch_critic import PatchCritic1D, PatchCritic2D


def test_critic_2D():
    critic = PatchCritic2D(nc=3, ndf=64, n_layers=2)
    t = torch.randn((64, 3, 32, 32))
    assert critic(t).shape == (64, 1, 6, 6)


def test_critic_1D():
    critic = PatchCritic1D(nc=2, ndf=64, n_layers=4)
    t = torch.randn((64, 2, 8192))
    assert critic(t).shape == (64, 1, 510)
