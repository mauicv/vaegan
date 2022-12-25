import pytest
import torch
from duct.model.critic import Critic


@pytest.mark.parametrize("res_blocks", [(0, 0, 0), (1, 1, 1), (1, 2, 0)])
def test_critic_2D(res_blocks):
    critic = Critic(
        3, 16, depth=3, 
        data_shape=(32, 32),
        res_blocks=res_blocks,
        attn_blocks=(0, 0, 1),
        downsample_block_type='image_block',
    )
    t = torch.randn((64, 3, 32, 32))
    assert critic(t).shape == (64, 1)


@pytest.mark.parametrize("res_blocks", [(0, 0, 0), (1, 1, 1), (1, 2, 0)])
def test_loss_2D(res_blocks):
    critic = Critic(
        3, 16, depth=3, 
        data_shape=(32, 32),
        res_blocks=res_blocks,
        attn_blocks=(0, 0, 1),
        downsample_block_type='image_block',
    )
    t1 = torch.randn((1, 3, 32, 32))
    t2 = torch.randn((1, 3, 32, 32))
    loss = critic.loss(t1, t2)
    assert loss.shape == (1, )
    loss = critic.loss(t1, t1)
    assert loss == 0


@pytest.mark.parametrize("res_blocks", [(0, 0, 0), (1, 1, 1), (1, 2, 0)])
def test_critic_1D(res_blocks):
    critic = Critic(
        2, 16, depth=3, 
        data_shape=(8192, ),
        res_blocks=res_blocks,
        attn_blocks=(0, 0, 1),
        downsample_block_type='audio_block',
    )
    t = torch.randn((64, 2, 8192))
    assert critic(t).shape == (64, 1)


@pytest.mark.parametrize("res_blocks", [(0, 0, 0), (1, 1, 1), (1, 2, 0)])
def test_loss_1D(res_blocks):
    critic = Critic(
        2, 16, depth=3, 
        data_shape=(8192, ),
        res_blocks=res_blocks,
        attn_blocks=(0, 0, 1),
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
        downsample_block_type='audio_block_v2',
    )
    t = torch.randn((64, 2, 8192))
    assert critic(t).shape == (64, 1)
