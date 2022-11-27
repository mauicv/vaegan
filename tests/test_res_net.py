import pytest
import torch
from duct.model.resnet import ResnetBlock


@pytest.mark.parametrize("conv_shortcut", [True, False])
def test_res_net(conv_shortcut):
    res_net_block = ResnetBlock(in_channels=64, out_channels=64, conv_shortcut=conv_shortcut, dropout=0.5)
    shape = (1, 64, 32, 32)
    t = torch.randn(shape)
    assert res_net_block(t).shape == shape
