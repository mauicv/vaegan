from duct.utils.config_mixin import ConfigMixin
from duct.model.autoencoders import NLLVarAutoEncoder2D, VarAutoEncoder2D, AutoEncoder2D, \
    VQVarAutoEncoder2D
from duct.model.critic import Critic
from duct.model.patch_critic import NLayerDiscriminator
from torch.optim import Adam
import pytest


class Experiment(ConfigMixin):
    name = 'test'

    def __init__(self, cfg):
        super().__init__(cfg)


def test_util_mixin_nll_vae(tmp_path):
    a = Experiment.from_file(path='./tests/test_configs/nll_vae.toml')
    assert a.vae.encoder.nc == 3
    assert isinstance(a.vae, NLLVarAutoEncoder2D)
    assert isinstance(a.vae_enc_opt, Adam)
    assert isinstance(a.vae_dec_opt, Adam)
    assert a.optimizers == ['vae_enc_opt', 'vae_dec_opt']

    path = tmp_path / 'model.pt'
    a.save_state(path)
    a.load_state(path)


def test_util_mixin_vq_vae(tmp_path):
    a = Experiment.from_file(path='./tests/test_configs/vq_vae.toml')
    print(a.vae)
    assert a.vae.encoder.nc == 3
    assert isinstance(a.vae, VQVarAutoEncoder2D)
    assert isinstance(a.vae_enc_opt, Adam)
    assert isinstance(a.vae_dec_opt, Adam)
    assert a.optimizers == ['vae_enc_opt', 'vae_dec_opt']

    path = tmp_path / 'model.pt'
    a.save_state(path)
    a.load_state(path)


def test_util_mixin_vae(tmp_path):
    a = Experiment.from_file(path='./tests/test_configs/vae.toml')
    assert a.vae.encoder.nc == 3
    assert isinstance(a.vae, VarAutoEncoder2D)
    with pytest.raises(KeyError):
        assert a.vae_opt

    path = tmp_path / 'model.pt'
    a.save_state(path)
    a.load_state(path)


def test_util_mixin_ae(tmp_path):
    a = Experiment.from_file(path='./tests/test_configs/ae.toml')
    assert a.ae.encoder.nc == 3
    assert isinstance(a.ae, AutoEncoder2D)
    assert isinstance(a.ae_opt, Adam)

    path = tmp_path / 'model.pt'
    a.save_state(path)
    a.load_state(path)


def test_util_mixin_critic(tmp_path):
    a = Experiment.from_file(path='./tests/test_configs/critic.toml')
    assert a.critic.encoder.nc == 3
    assert isinstance(a.critic, Critic)
    assert isinstance(a.critic_opt, Adam)

    path = tmp_path / 'model.pt'
    a.save_state(path)
    a.load_state(path)


def test_util_mixin_patch_critic(tmp_path):
    a = Experiment.from_file(path='./tests/test_configs/patch_critic.toml')
    assert isinstance(a.patch_critic, NLayerDiscriminator)
    assert isinstance(a.patch_critic_opt, Adam)

    path = tmp_path / 'model.pt'
    a.save_state(path)
    a.load_state(path)


def test_util_mixin_from_toml(tmp_path):
    toml_str = '''
    [vae]
    class = 'NLLVarAutoEncoder2D'
    nc = 3
    ndf = 16
    img_shape = [ 128, 128,]
    depth = 6
    res_blocks = [0, 0, 0, 0, 0, 0]

    [[vae.opt_cfgs]]
    class='Adam'
    name='vae_enc_opt'
    parameter_set='encoder_params'
    lr = 0.0005

    [[vae.opt_cfgs]]
    class='Adam'
    name='vae_dec_opt'
    parameter_set='decoder_params'
    lr = 0.0005
    '''

    a = Experiment.from_toml(toml_str)
    assert a.vae.encoder.nc == 3
    assert isinstance(a.vae, NLLVarAutoEncoder2D)
    assert isinstance(a.vae_enc_opt, Adam)
    assert isinstance(a.vae_dec_opt, Adam)
    assert a.optimizers == ['vae_enc_opt', 'vae_dec_opt']

    path = tmp_path / 'model.pt'
    a.save_state(path)
    a.load_state(path)
