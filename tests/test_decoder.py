from model.decoder import Decoder, UpSampleBatchConvBlock
import torch


def test_us_block():
    t = torch.randn((64, 16, 32, 32))
    downsample_block = UpSampleBatchConvBlock(16, 3)
    dst = downsample_block(t)
    assert dst.shape == (64, 3, 64, 64)


def test_decoder():
    decoder = Decoder(3, 16, depth=3,
                      img_shape=(32, 32),
                      latent_dim=10)
    t = torch.randn((64, 10))
    x = decoder(t)
    assert x.shape == (64, 3, 32, 32)