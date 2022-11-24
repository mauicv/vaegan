import torch
from model.critic import Critic


def test_critic():
    critic = Critic(3, 16, depth=3, img_shape=(32, 32))
    t = torch.randn((64, 3, 32, 32))
    assert critic(t).shape == (64, 1)


def test_loss():
    critic = Critic(3, 16, depth=3, img_shape=(32, 32))
    t1 = torch.randn((1, 3, 32, 32))
    t2 = torch.randn((1, 3, 32, 32))
    loss = critic.loss(t1, t2)
    assert loss.shape == (1, )
    loss = critic.loss(t1, t1)
    assert loss == 0