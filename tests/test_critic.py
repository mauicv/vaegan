import pytest
import torch
from duct.model.critic import Critic


@pytest.mark.parametrize("res_blocks", [(0, 0, 0), (1, 1, 1), (1, 2, 0)])
def test_critic(res_blocks):
    critic = Critic(
        3, 16, depth=3, 
        img_shape=(32, 32),
        res_blocks=res_blocks
    )
    t = torch.randn((64, 3, 32, 32))
    assert critic(t).shape == (64, 1)


@pytest.mark.parametrize("res_blocks", [(0, 0, 0), (1, 1, 1), (1, 2, 0)])
def test_loss(res_blocks):
    critic = Critic(
        3, 16, depth=3, 
        img_shape=(32, 32),
        res_blocks=res_blocks
    )
    t1 = torch.randn((1, 3, 32, 32))
    t2 = torch.randn((1, 3, 32, 32))
    loss = critic.loss(t1, t2)
    assert loss.shape == (1, )
    loss = critic.loss(t1, t1)
    assert loss == 0