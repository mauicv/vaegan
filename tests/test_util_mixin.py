from duct.utils.util_mixin import UtilMixin
from duct.model.autoencoders import NLLVarAutoEncoder, VarAutoEncoder, AutoEncoder
from duct.model.critic import Critic
from duct.model.patch_critic import NLayerDiscriminator
from torch.optim import Adam
import pytest


class A(UtilMixin):
        def __init__(self, cfg):
            super().__init__(cfg)


def test_util_mixin_nll_vae(tmp_path):
    a = A.from_file(path='./tests/test_configs/nll_vae.toml')
    assert a.vae['nc'] == 3
    a.load()
    assert a.vae.encoder.nc == 3
    assert isinstance(a.vae, NLLVarAutoEncoder)
    assert isinstance(a.vae_enc_opt, Adam)
    assert isinstance(a.vae_dec_opt, Adam)
    assert a.optimizers == ['vae_enc_opt', 'vae_dec_opt']

    path = tmp_path / 'model.pt'
    a.save_state(path)
    a.load_state(path)



def test_util_mixin_vae(tmp_path):
    a = A.from_file(path='./tests/test_configs/vae.toml')
    assert a.vae['nc'] == 3
    a.load()
    assert a.vae.encoder.nc == 3
    assert isinstance(a.vae, VarAutoEncoder)
    with pytest.raises(KeyError):
        assert a.vae_opt

    path = tmp_path / 'model.pt'
    a.save_state(path)
    a.load_state(path)


def test_util_mixin_ae(tmp_path):
    a = A.from_file(path='./tests/test_configs/ae.toml')
    assert a.ae['nc'] == 3
    a.load()
    assert a.ae.encoder.nc == 3
    assert isinstance(a.ae, AutoEncoder)
    assert isinstance(a.ae_opt, Adam)

    path = tmp_path / 'model.pt'
    a.save_state(path)
    a.load_state(path)


def test_util_mixin_critic(tmp_path):
    a = A.from_file(path='./tests/test_configs/critic.toml')
    assert a.critic['nc'] == 3
    a.load()
    assert a.critic.encoder.nc == 3
    assert isinstance(a.critic, Critic)
    assert isinstance(a.critic_opt, Adam)

    path = tmp_path / 'model.pt'
    a.save_state(path)
    a.load_state(path)


def test_util_mixin_patch_critic(tmp_path):
    a = A.from_file(path='./tests/test_configs/patch_critic.toml')
    assert a.patch_critic['nc'] == 3
    a.load()
    assert isinstance(a.patch_critic, NLayerDiscriminator)
    assert isinstance(a.patch_critic_opt, Adam)

    path = tmp_path / 'model.pt'
    a.save_state(path)
    a.load_state(path)
