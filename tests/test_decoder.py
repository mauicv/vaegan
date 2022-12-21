import pytest
from duct.model.decoder import Decoder, UpSampleBatchConvBlock
import torch


def test_us_block():
    t = torch.randn((64, 16, 32, 32))
    downsample_block = UpSampleBatchConvBlock(16, 3)
    dst = downsample_block(t)
    assert dst.shape == (64, 3, 64, 64)


@pytest.mark.parametrize("res_blocks", [(0, 0, 0), (1, 1, 1), (1, 2, 1)])
def test_decoder(res_blocks):
    decoder = Decoder(3, 16, depth=3, data_shape=(32, 32), res_blocks=res_blocks)
    t = torch.randn((64, 128, 8, 8))
    x = decoder(t)
    assert x.shape == (64, 3, 64, 64)