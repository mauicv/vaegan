from utils.saving import setup_models
import torch


def test_saving(tmp_path):
    new, save, load = setup_models()
    models = new()
    for key in ['vae', 'critic', 'patch_critic', 'critic_opt',
                'enc_opt', 'dec_opt', 'patch_critic_opt']:
        key in models.keys()

    for model in models.values():
        if isinstance(model, torch.nn.Module):
            model.train(False)

    vae = models['vae']
    critic = models['critic']
    patch_critic = models['patch_critic']
    # critic_opt = models['critic_opt']
    # patch_critic_opt = models['patch_critic_opt']
    # enc_opt = models['enc_opt']
    # dec_opt = models['dec_opt']

    t = torch.randn((64, 3, 128, 128))
    _, mu1, _ = vae.encode(t)
    _, mup, _ = vae(t)
    assert mu1.shape == mup.shape
    tp = vae.decode(mup)
    t1 = vae.decode(mu1)
    assert tp.shape == t1.shape

    assert t1.shape == (64, 3, 128, 128)
    l1 = critic(t1)
    pl1 = patch_critic(t1)
    assert l1.shape == (64, 1)
    assert pl1.shape == (64, 1, 14, 14)

    save(**models)
    models = load()
    for key in ['vae', 'critic', 'patch_critic', 'critic_opt',
                'enc_opt', 'dec_opt', 'patch_critic_opt']:
        key in models.keys()

    for model in models.values():
        if isinstance(model, torch.nn.Module):
            model.train(False)

    _, mu1, _ = vae.encode(t)
    t2 = vae.decode(mu1)
    assert t2.shape == (64, 3, 128, 128)
    l2 = critic(t2)
    pl2 = patch_critic(t2)
    assert l2.shape == (64, 1)
    assert pl2.shape == (64, 1, 14, 14)

    assert (l1.detach().numpy() == l2.detach().numpy()).all()
    assert (pl1.detach().numpy() == pl2.detach().numpy()).all()
    assert (t1.detach().numpy() == t2.detach().numpy()).all()
