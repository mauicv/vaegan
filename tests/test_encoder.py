from model.encoder import Encoder, DownSampleBlock
import torch


def test_ds_block():
    t = torch.randn((64, 3, 128, 128))
    downsample_block = DownSampleBlock(3, 16)
    dst = downsample_block(t)
    assert dst.shape == (64, 16, 64, 64)


def test_encoder():
    encoder = Encoder(3, 16, depth=3,
                      img_shape=(32, 32),
                      norm_type='instance')
    t = torch.randn((64, 3, 32, 32))
    x = encoder(t)
    assert x.shape == (64, 4096)
