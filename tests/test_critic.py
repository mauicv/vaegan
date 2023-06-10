import pytest
import torch
from duct.model.critic import Critic, MultiResCritic, SpectralCritic, MultiScaleSpectralCritic


@pytest.mark.parametrize("res_blocks", [(0, 0, 0), (1, 2, 0)])
def test_critic_2D(res_blocks):
    critic = Critic(
        3, 16, depth=3, 
        data_shape=(32, 32),
        res_blocks=res_blocks,
        attn_blocks=(0, 0, 1),
        ch_mult=(2, 2, 2),
        downsample_block_type='image_block',
    )
    t = torch.randn((64, 3, 32, 32))
    assert critic(t).shape == (64, 1)


@pytest.mark.parametrize("res_blocks", [(0, 0, 0), (1, 2, 0)])
def test_loss_2D(res_blocks):
    critic = Critic(
        3, 16, depth=3, 
        data_shape=(32, 32),
        res_blocks=res_blocks,
        attn_blocks=(0, 0, 1),
        ch_mult=(2, 2, 2),
        downsample_block_type='image_block',
    )
    t1 = torch.randn((1, 3, 32, 32))
    t2 = torch.randn((1, 3, 32, 32))
    loss = critic.loss(t1, t2)
    assert loss.shape == (1, )
    loss = critic.loss(t1, t1)
    assert loss == 0


@pytest.mark.parametrize("res_blocks", [(0, 0, 0), (1, 2, 0)])
def test_critic_1D(res_blocks):
    critic = Critic(
        2, 16, depth=3, 
        data_shape=(8192, ),
        res_blocks=res_blocks,
        attn_blocks=(0, 0, 1),
        ch_mult=(2, 2, 2),
        downsample_block_type='audio_block',
    )
    t = torch.randn((64, 2, 8192))
    assert critic(t).shape == (64, 1)


@pytest.mark.parametrize("res_blocks", [(0, 0, 0), (1, 2, 0)])
def test_loss_1D(res_blocks):
    critic = Critic(
        2, 16, depth=3, 
        data_shape=(8192, ),
        res_blocks=res_blocks,
        attn_blocks=(0, 0, 1),
        ch_mult=(2, 2, 2),
        downsample_block_type='audio_block',
    )
    t1 = torch.randn((1, 2, 8192))
    t2 = torch.randn((1, 2, 8192))
    loss = critic.loss(t1, t2)
    assert loss.shape == (1, )
    loss = critic.loss(t1, t1)
    assert loss == 0


@pytest.mark.parametrize("res_blocks", [(0, 0, 0, 0, 0, 0)])
def test_critic_1D_aud(res_blocks):
    critic = Critic(
        2, 16, depth=6, 
        data_shape=(8192, ),
        res_blocks=res_blocks,
        attn_blocks=(0, 0, 0, 0, 0, 1),
        ch_mult=(2, 2, 2, 2, 2, 2),
        downsample_block_type='audio_block',
    )
    t = torch.randn((64, 2, 8192))
    assert critic(t).shape == (64, 1)


def test_critic_1D_aud_v2():
    critic = Critic(
        2, 16, depth=8, 
        data_shape=(8192, ),
        res_blocks=(0, 0, 0, 0, 0, 0, 0, 0),
        attn_blocks=(0, 0, 0, 0, 0, 0, 0, 0),
        ch_mult=(2, 2, 2, 2, 2, 2, 2, 2),
        downsample_block_type='audio_block_v2',
    )
    t = torch.randn((64, 2, 8192))
    assert critic(t).shape == (64, 1)


@pytest.mark.parametrize("res_blocks", [(0, 0, 0), (1, 2, 0)])
def test_multi_res_critic_1D(res_blocks):
    critic = MultiResCritic(
        2, 16, depth=3, 
        data_shape=(8192, ),
        res_blocks=res_blocks,
        attn_blocks=(0, 0, 1),
        ch_mult=(2, 2, 2),
        downsample_block_type='audio_block',
        num_resolutions=3,
        patch=True
    )
    t = torch.randn((64, 2, 8192))
    results = critic(t)
    for (_, val), patches in zip(results.items(), (128, 64, 32)):
        assert val.shape == (64, patches)


@pytest.mark.parametrize("res_blocks", [(0, 0, 0), (1, 2, 0)])
def test_multi_res_loss_1D(res_blocks):
    critic = MultiResCritic(
        2, 16, depth=3, 
        data_shape=(8192, ),
        res_blocks=res_blocks,
        attn_blocks=(0, 0, 1),
        ch_mult=(2, 2, 2),
        downsample_block_type='audio_block',
        num_resolutions=3
    )
    t1 = torch.randn((1, 2, 8192))
    t2 = torch.randn((1, 2, 8192))
    loss = critic.loss(t1, t2)
    assert loss > 0
    loss = critic.loss(t1, t1)
    assert loss == 0


def test_spectral_critic():
    critic = SpectralCritic(
        nc=1,
        ndf=32,
        depth=3,
        patch=True,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
    )
    t = torch.randn(1, 1, 24000)
    logits, fmap = critic(t)
    assert logits.shape == (1, 1, 86, 61)
    assert len(fmap) == 4


def test_multi_res_spectral_critic():
    critic = MultiScaleSpectralCritic(
        nc=1,
        ndf=32,
        depth=3,
        patch=True,
        n_ffts=[2048, 1024, 512, 256, 128],
        hop_lengths=[512, 256, 128, 64, 32],
        win_lengths=[2048, 1024, 512, 256, 128],
        normalized=True,
    )
    t = torch.randn(2, 1, 24000)
    logits, fmaps = critic(t)
    assert all([len(l.shape)==4 for l in logits])
    for fmap_set in fmaps:
        assert all([len(l.shape)==4 for l in fmap_set])

    
def test_multi_res_spectral_critic_loss():
    critic = MultiScaleSpectralCritic(
        nc=1,
        ndf=32,
        depth=3,
        patch=True,
        n_ffts=[2048, 1024, 512, 256, 128],
        hop_lengths=[512, 256, 128, 64, 32],
        win_lengths=[2048, 1024, 512, 256, 128],
        normalized=True,
    )
    t1 = torch.randn(2, 1, 24000)
    t2 = torch.randn(2, 1, 24000)
    _, fmaps_1 = critic(t1)
    _, fmaps_2 = critic(t2)
    loss = critic.relative_feature_loss(fmaps_1, fmaps_2)
    assert loss.shape == (2, )
    assert loss.sum() > 0

    t = torch.randn(1, 1, 24000)
    _, fmaps = critic(t)
    loss = critic.relative_feature_loss(fmaps, fmaps)
    assert loss.shape == (1, )
    assert loss == 0