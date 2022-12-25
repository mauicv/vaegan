
import pytest
from duct.model.encoder import Encoder, DownSampleBlock
import torch


@pytest.mark.parametrize("dimensions", [(32, 32), (64, 64), (128, 128)])
def test_ds_block_image(dimensions):
    t = torch.randn((64, 3, *dimensions))
    downsample_block = DownSampleBlock.image_block(3, 16)
    dst = downsample_block(t)
    assert dst.shape == (64, 16, *(int(d/2) for d in dimensions))


@pytest.mark.parametrize("audio_length", [64, 128, 8192])
def test_ds_block_audio(audio_length):
    t = torch.randn((64, 2, audio_length))
    downsample_block = DownSampleBlock.audio_block(2, 16)
    dst = downsample_block(t)
    assert dst.shape == (64, 16, int(audio_length/4))


@pytest.mark.parametrize("res_blocks", [(0, 0, 0), (1, 1, 1), (1, 2, 0)])
def test_encoder_image(res_blocks):
    encoder = Encoder(3, 16, depth=3,
                      data_shape=(32, 32),
                      res_blocks=res_blocks,
                      attn_blocks=(0, 0, 1))
    t = torch.randn((64, 3, 32, 32))
    x = encoder(t)
    assert x.shape == (64, 128, 4, 4)


@pytest.mark.parametrize("res_blocks", [(0, 0, 0, 0), (1, 1, 1, 0), (1, 2, 0, 1)])
def test_encoder_audio(res_blocks):
    encoder = Encoder(2, 16, depth=4,
                      data_shape=(8192, ),
                      res_blocks=res_blocks,
                      attn_blocks=(0, 0, 1, 0),
                      downsample_block_type='audio_block',)
    t = torch.randn((64, 2, 8192))
    x = encoder(t)
    assert x.shape == (64, 256, 32)


def test_encoder_audio_v2():
    encoder = Encoder(2, 16, depth=3,
                      data_shape=(8192, ),
                      res_blocks=(0, 0, 0),
                      attn_blocks=(0, 0, 0),
                      downsample_block_type='audio_block_v2',)
    t = torch.randn((64, 2, 8192))
    x = encoder(t)
    assert x.shape == (64, 128, 1024)

