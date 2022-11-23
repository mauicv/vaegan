from utils.saving import setup_models

def test_saving(tmp_path):
    new, save, load = setup_models()
    models = new()
    for key in ['vae', 'critic', 'patch_critic', 'critic_opt',
                'enc_opt', 'dec_opt', 'patch_critic_opt']:
        key in models.keys()

    # vae = models['vae']
    # critic = models['critic']
    # patch_critic = models['patch_critic']
    # critic_opt = models['critic_opt']
    # patch_critic_opt = models['patch_critic_opt']
    # enc_opt = models['enc_opt']
    # dec_opt = models['dec_opt']

    save(**models)
    models = load()
    for key in ['vae', 'critic', 'patch_critic', 'critic_opt',
                'enc_opt', 'dec_opt', 'patch_critic_opt']:
        key in models.keys()
