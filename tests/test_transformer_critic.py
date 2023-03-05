import pytest
import torch
from torch.optim import Adam
from duct.model.transformer_critic import TransformerCritic
from duct.utils.config_mixin import ConfigMixin


class Experiment(ConfigMixin):
    name = 'test'

    def __init__(self, cfg):
        super().__init__(cfg)



@pytest.mark.parametrize("res_blocks", [(1, 1, 0), (1, 2, 0)])
@pytest.mark.parametrize("num_transformer_blocks", [2, 3])
def test_transformer_critic_2D(res_blocks, num_transformer_blocks):
    transformer_critic = TransformerCritic(
        3, 16, depth=3, 
        data_shape=(32, 32),
        res_blocks=res_blocks,
        attn_blocks=(0, 0, 1),
        downsample_block_type='image_block',
        num_transformer_blocks=num_transformer_blocks,
    )
    t = torch.randn((64, 3, 32, 32))
    assert transformer_critic(t).shape == (1, 64, 64)


def test_transformer_critic_config(tmp_path):
    toml_str = '''
    [critic]
    class = 'TransformerCritic'
    nc = 3
    ndf = 16
    data_shape = [ 128, 128,]
    depth = 6
    res_blocks = [0, 1, 0, 1, 1, 0]
    attn_blocks = [0, 0, 0, 0, 0, 1]
    num_transformer_blocks = 2

    [[critic.opt_cfgs]]
    class='Adam'
    name='critic_opt'
    lr = 0.0005
    '''

    a = Experiment.from_toml(toml_str)
    assert a.critic.encoder.nc == 3
    assert isinstance(a.critic, TransformerCritic)
    assert len(a.critic.transformer_blocks) == 2
    assert isinstance(a.critic_opt, Adam)

    path = tmp_path / 'model.pt'
    a.save_state(path)
    a.load_state(path)
