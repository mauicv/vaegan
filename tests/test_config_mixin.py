from duct.utils.config_mixin import ConfigMixin
from duct.model.autoencoders import NLLVarAutoEncoder, VarAutoEncoder, AutoEncoder, \
    VQVarAutoEncoder
from duct.model.critic import Critic, MultiResCritic, SpectralCritic
from duct.model.patch_critic import PatchCritic1D, PatchCritic2D
from duct.model.transformer.model import Transformer, RelEmbTransformer
from torch.optim import Adam, AdamW
from freezegun import freeze_time
import os


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
    a.save_state(tmp_path)
    a.load_state(tmp_path)


def test_util_mixin_vq_vae(tmp_path):
    a = Experiment.from_file(path='./tests/test_configs/vq_vae.toml')
    assert a.vae.encoder.nc == 3
    assert isinstance(a.vae, VQVarAutoEncoder)
    assert isinstance(a.vae_enc_opt, Adam)
    assert isinstance(a.vae_dec_opt, Adam)
    assert a.optimizers == ['vae_enc_opt', 'vae_dec_opt']
    a.save_state(tmp_path)
    a.load_state(tmp_path)


def test_util_mixin_vae(tmp_path):
    a = Experiment.from_file(path='./tests/test_configs/vae.toml')
    assert a.vae.encoder.nc == 3
    assert isinstance(a.vae, VarAutoEncoder)
    assert a.vae_opt == None
    a.save_state(tmp_path)
    a.load_state(tmp_path)


def test_util_mixin_ae(tmp_path):
    a = Experiment.from_file(path='./tests/test_configs/ae.toml')
    assert a.ae.encoder.nc == 3
    assert isinstance(a.ae, AutoEncoder)
    assert isinstance(a.ae_opt, Adam)
    a.save_state(tmp_path)
    a.load_state(tmp_path)


def test_util_mixin_critic(tmp_path):
    a = Experiment.from_file(path='./tests/test_configs/critic.toml')
    assert a.critic.encoder.nc == 3
    assert isinstance(a.critic, Critic)
    assert isinstance(a.critic_opt, Adam)
    a.save_state(tmp_path)
    a.load_state(tmp_path)


def test_util_mixin_patch_critic(tmp_path):
    a = Experiment.from_file(path='./tests/test_configs/patch_critic_2d.toml')
    assert isinstance(a.patch_critic, PatchCritic2D)
    assert isinstance(a.patch_critic_opt, Adam)
    a.save_state(tmp_path)
    a.load_state(tmp_path)


def test_util_mixin_patch_critic(tmp_path):
    a = Experiment.from_file(path='./tests/test_configs/patch_critic_1d.toml')
    assert isinstance(a.patch_critic, PatchCritic1D)
    assert isinstance(a.patch_critic_opt, Adam)
    a.save_state(tmp_path)
    a.load_state(tmp_path)


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
    ch_mult = [2, 2, 2, 2, 2, 2]

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
    a.save_state(tmp_path)
    a.load_state(tmp_path)


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
    ch_mult = [2, 2, 2, 2, 2, 2]
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
    a.save_state(tmp_path)
    a.load_state(tmp_path)


def test_util_mixin_transformer(tmp_path):
    exp = Experiment.from_file(path='./tests/test_configs/transformer.toml')
    assert isinstance(exp.transformer, Transformer)
    assert isinstance(exp.transformer_opt, AdamW)

    assert exp.transformer.n_heads == 8
    assert exp.transformer.emb_dim == 256
    assert exp.transformer.emb_num == 10
    assert exp.transformer.depth == 5
    exp.save_state(tmp_path)
    exp.load_state(tmp_path)


def test_util_mixin_transformer(tmp_path):
    exp = Experiment.from_file(path='./tests/test_configs/rel_emb_transformer.toml')
    assert isinstance(exp.transformer, RelEmbTransformer)
    assert isinstance(exp.transformer_opt, AdamW)

    assert exp.transformer.n_heads == 8
    assert exp.transformer.emb_dim == 256
    assert exp.transformer.emb_num == 10
    assert exp.transformer.depth == 5

    exp.save_state(tmp_path)
    exp.load_state(tmp_path)


@freeze_time("Jan 14th, 2020", auto_tick_seconds=15)
def test_util_mixin_num_replicas(tmp_path):
    toml_str = """
    num_saved_replicas = 3

    [transformer]
    class = 'Transformer'
    n_heads = 8
    emb_dim = 256
    emb_num = 10
    depth = 5
    block_size = 128

    [[transformer.opt_cfgs]]
    class = 'AdamW'
    name = 'transformer_opt'
    parameter_set = 'get_parameter_groups'
    lr = 0.0005
    """
    exp = Experiment.from_toml(toml_str)
    for _ in range(5):
        exp.save_state(tmp_path)
    assert '2020-01-14|00:01:00.pt' == \
        str(exp._get_replica_path(tmp_path, 'latest')).split('/')[-1]
    assert len(os.listdir(tmp_path)) == 3
    exp.load_state(tmp_path)


def test_util_mixin_multi_res_critic(tmp_path):
    a = Experiment.from_file(path='./tests/test_configs/multi_res_critic.toml')
    assert len(a.critic.critics) == 3
    assert isinstance(a.critic, MultiResCritic)
    assert isinstance(a.critic_opt, Adam)
    a.save_state(tmp_path)
    a.load_state(tmp_path) 


def test_util_mixin_spectral_critic(tmp_path):
    a = Experiment.from_file(path='./tests/test_configs/spectral_critic.toml')
    assert isinstance(a.critic, SpectralCritic)
    assert isinstance(a.critic_opt, Adam)
    a.save_state(tmp_path)
    a.load_state(tmp_path) 