from duct.utils.config_mixin import ConfigMixin
import torch

class Experiment(ConfigMixin):
    name = 'test'


def test_saving(tmp_path):
    test_class = Experiment.from_file(path='./tests/test_configs/config_2d.toml')
    assert test_class.models == ['vae', 'critic', 'patch_critic']
    assert test_class.optimizers == ['vae_enc_opt', 'vae_dec_opt', 'critic_opt', 'patch_critic_opt']

    models = [getattr(test_class, model) for model in test_class.models]
    for model in models:
        if isinstance(model, torch.nn.Module):
            model.train(False)

    t = torch.randn((64, 3, 128, 128))

    _, mu1, _ = test_class.vae.encode(t)
    _, mup, _ = test_class.vae(t)
    assert mu1.shape == mup.shape
    tp = test_class.vae.decode(mup)
    t1 = test_class.vae.decode(mu1)
    assert tp.shape == t1.shape

    assert t1.shape == (64, 3, 128, 128)
    l1 = test_class.critic(t1)
    pl1 = test_class.patch_critic(t1)
    assert l1.shape == (64, 1)
    assert pl1.shape == (64, 1, 14, 14)

    test_class.save_state(tmp_path / 'model.pt')
    test_class = None
    models = None

    test_class2 = Experiment.from_file(path='./tests/test_configs/config_2d.toml')
    assert test_class2.models == ['vae', 'critic', 'patch_critic']
    assert test_class2.optimizers == ['vae_enc_opt', 'vae_dec_opt', 'critic_opt', 'patch_critic_opt']

    test_class2.load_state(tmp_path / 'model.pt')
    models = [getattr(test_class2, model) for model in test_class2.models]
    for model in models:
        if isinstance(model, torch.nn.Module):
            model.train(False)

    _, mu1, _ = test_class2.vae.encode(t)
    t2 = test_class2.vae.decode(mu1)
    assert t2.shape == (64, 3, 128, 128)
    l2 = test_class2.critic(t2)
    pl2 = test_class2.patch_critic(t2)
    assert l2.shape == (64, 1)
    assert pl2.shape == (64, 1, 14, 14)

    assert (l1.detach().numpy() == l2.detach().numpy()).all()
    assert (pl1.detach().numpy() == pl2.detach().numpy()).all()
    assert (t1.detach().numpy() == t2.detach().numpy()).all()
