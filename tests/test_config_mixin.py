from duct.utils.config_mixin import ConfigMixin
from duct.model.autoencoders import NLLVarAutoEncoder, VarAutoEncoder, AutoEncoder, \
    VQVarAutoEncoder
from duct.model.critic import Critic
from duct.model.patch_critic import NLayerDiscriminator
from duct.model.transformer.model import Transformer
from torch.optim import Adam, AdamW
import pytest


class Experiment(ConfigMixin):
    name = 'test'

    def __init__(self, cfg):
        super().__init__(cfg)


def test_util_mixin_nll_vae(tmp_path):
    a = Experiment.from_file(path='./tests/test_configs/nll_vae.toml')
    assert a.vae.encoder.nc == 3
    assert isinstance(a.vae, NLLVarAutoEncoder)
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
    assert isinstance(a.vae, VQVarAutoEncoder)
    assert isinstance(a.vae_enc_opt, Adam)
    assert isinstance(a.vae_dec_opt, Adam)
    assert a.optimizers == ['vae_enc_opt', 'vae_dec_opt']

    path = tmp_path / 'model.pt'
    a.save_state(path)
    a.load_state(path)


def test_util_mixin_vae(tmp_path):
    a = Experiment.from_file(path='./tests/test_configs/vae.toml')
    assert a.vae.encoder.nc == 3
    assert isinstance(a.vae, VarAutoEncoder)
    with pytest.raises(KeyError):
        assert a.vae_opt

    path = tmp_path / 'model.pt'
    a.save_state(path)
    a.load_state(path)


def test_util_mixin_ae(tmp_path):
    a = Experiment.from_file(path='./tests/test_configs/ae.toml')
    assert a.ae.encoder.nc == 3
    assert isinstance(a.ae, AutoEncoder)
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
    class = 'NLLVarAutoEncoder'
    nc = 3
    ndf = 16
    data_shape = [ 128, 128,]
    depth = 6
    res_blocks = [0, 0, 0, 0, 0, 0]
    attn_blocks = [0, 0, 0, 0, 0, 1]

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
    assert isinstance(a.vae, NLLVarAutoEncoder)
    assert isinstance(a.vae_enc_opt, Adam)
    assert isinstance(a.vae_dec_opt, Adam)
    assert a.optimizers == ['vae_enc_opt', 'vae_dec_opt']

    path = tmp_path / 'model.pt'
    a.save_state(path)
    a.load_state(path)


def test_1d_vq_config(tmp_path):
    toml_str = '''
    [vae]
    class = 'VQVarAutoEncoder'
    nc = 2
    ndf = 16
    data_shape = [ 128, ]
    depth = 6
    res_blocks = [0, 0, 0, 0, 0, 0]
    attn_blocks = [0, 0, 0, 0, 0, 0]
    upsample_block_type = 'audio_block'
    downsample_block_type = 'audio_block'

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
    assert a.vae.encoder.nc == 2
    assert isinstance(a.vae, VQVarAutoEncoder)
    assert isinstance(a.vae_enc_opt, Adam)
    assert isinstance(a.vae_dec_opt, Adam)
    assert a.optimizers == ['vae_enc_opt', 'vae_dec_opt']

    path = tmp_path / 'model.pt'
    a.save_state(path)
    a.load_state(path)


def test_util_mixin_transformer(tmp_path):
    exp = Experiment.from_file(path='./tests/test_configs/transformer.toml')
    assert isinstance(exp.transformer, Transformer)
    assert isinstance(exp.transformer_opt, AdamW)

    assert exp.transformer.n_heads == 8
    assert exp.transformer.emb_dim == 256
    assert exp.transformer.emb_num == 10
    assert exp.transformer.depth == 5

    path = tmp_path / 'model.pt'
    exp.save_state(path)
    exp.load_state(path)