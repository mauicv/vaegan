from duct.utils.config_mixin import ConfigMixin
from duct.utils.logging_mixin import LoggingMixin
from duct.utils.save_imgs import save_img_pairs
import torch


def test_saving(tmp_path):
    class Experiment(ConfigMixin, LoggingMixin):
        img_save_hook = save_img_pairs
        headers = ['vae_enc_loss', 'vae_dec_loss', 'critic_loss', 'patch_critic_loss']
        name = str(tmp_path)

    test_class = Experiment.from_file(path='./tests/test_configs/config.toml')
    test_class.setup_logs()

    assert test_class.models == ['vae', 'critic', 'patch_critic']
    assert test_class.optimizers == ['vae_enc_opt', 'vae_dec_opt', 'critic_opt', 'patch_critic_opt']

    imgs_1 = torch.randn(6, 3, 128, 128)
    imgs_2 = test_class.vae.call(imgs_1)
    assert imgs_1.shape == imgs_2.shape

    test_class.save_state(tmp_path / 'model.pt')
    test_class.save_imgs(imgs_1, imgs_2)