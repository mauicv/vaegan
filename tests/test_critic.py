# import torch
# from model.critic import Critic, discriminator_block


# def test_critic():
#     critic = Critic(3, 16, depth=3, img_shape=(32, 32))
#     t = torch.randn((64, 3, 32, 32))
#     assert critic(t).shape == (64, 1)


# def test_loss():
#     critic = Critic(3, 16, depth=3, img_shape=(32, 32))
#     t1 = torch.randn((1, 3, 32, 32))
#     t2 = torch.randn((1, 3, 32, 32))
#     loss = critic.loss(t1, t2)
#     assert loss.shape == (1, )
#     assert 0 < loss < 1


# def test_disc_block():
#     disc_layer = discriminator_block(16, 32)
#     t = torch.randn((64, 16, 32, 32))
#     disc_layer(t).shape == (64, 32, 16, 16)

#     disc_layer = discriminator_block(16, 32, down_sample=False)
#     t = torch.randn((64, 16, 32, 32))
#     disc_layer(t).shape == (64, 16, 32, 32)