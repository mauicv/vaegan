
import pytest
from duct.model.encoder import Encoder, DownSampleInstanceConv2dBlock
import torch


def test_ds_block():
    t = torch.randn((64, 3, 128, 128))
    downsample_block = DownSampleInstanceConv2dBlock(3, 16)
    dst = downsample_block(t)
    assert dst.shape == (64, 16, 64, 64)


@pytest.mark.parametrize("res_blocks", [(0, 0, 0), (1, 1, 1), (1, 2, 0)])
def test_encoder(res_blocks):
    encoder = Encoder(3, 16, depth=3,
                      img_shape=(32, 32),
                      res_blocks=res_blocks)
    t = torch.randn((64, 3, 32, 32))
    x = encoder(t)
    assert x.shape == (64, 128, 4, 4)