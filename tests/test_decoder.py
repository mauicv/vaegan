import pytest
from duct.model.decoder import Decoder, UpSampleBlock
import torch


def test_us_block_image():
    t = torch.randn((64, 16, 32, 32))
    downsample_block = UpSampleBlock.image_block(16, 3)
    dst = downsample_block(t)
    assert dst.shape == (64, 3, 64, 64)


def test_us_block_audio():
    t = torch.randn((64, 16, 32))
    downsample_block = UpSampleBlock.audio_block(16, 2)
    dst = downsample_block(t)
    assert dst.shape == (64, 2, 32*4)


@pytest.mark.parametrize("res_blocks", [(0, 0, 0), (1, 1, 1), (1, 2, 1)])
def test_decoder_image(res_blocks):
    decoder = Decoder(3, 16, depth=3, data_shape=(32, 32), 
                      res_blocks=res_blocks, attn_blocks=(0, 0, 1),)
    t = torch.randn((64, 128, 8, 8))
    x = decoder(t)
    assert x.shape == (64, 3, 64, 64)

@pytest.mark.parametrize("res_blocks", [(0, 0, 0, 0), (1, 1, 1, 1), (1, 2, 1, 0)])
def test_decoder_audio(res_blocks):
    decoder = Decoder(2, 16, depth=4, data_shape=(8192, ), 
                      res_blocks=res_blocks, 
                      attn_blocks=(0, 0, 1, 0),
                      upsample_block_type='audio_block')
    t = torch.randn((64, 256, 32))
    x = decoder(t)
    assert x.shape == (64, 2, 8192)


def test_decoder_audio_v2():
    decoder = Decoder(2, 16, depth=3, data_shape=(8192, ), 
                      res_blocks=(0, 0, 0),
                      attn_blocks=(0, 0, 0),
                      upsample_block_type='audio_block_v2')
    t = torch.randn((64, 128, 1024))
    x = decoder(t)
    assert x.shape == (64, 2, 8192)
