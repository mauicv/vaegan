from model.autoencoders import VarAutoEncoder
from model.critic import Critic
from model.patch_critic import NLayerDiscriminator
import torch
from pathlib import Path
from utils.config import load_config


def setup_models():
    cfg = load_config()
    PATH = Path('./saved')
    PATH.mkdir(exist_ok=True)
    MODEL_PATH = f'{str(PATH)}/model.pt'

    def new():
        critic = Critic(**cfg['CRITIC_PARAMS'])
        vae = VarAutoEncoder(**cfg['VAE_PARAMS'])
        patch_critic = NLayerDiscriminator(**cfg['PATCH_CRITIC_PARAMS'])
        critic_opt = torch.optim.Adam(critic.parameters(), **cfg['CRITIC_OPT_PARAMS'])
        enc_opt = torch.optim.Adam(vae.encoder_params(), **cfg['ENC_OPT_PARAMS'])
        dec_opt = torch.optim.Adam(vae.decoder_params(), **cfg['DEC_OPT_PARAMS'])
        patch_critic_opt = torch.optim.Adam(patch_critic.parameters(), **cfg['PATCH_CRITIC_OPT_PARAMS'])

        if cfg['cuda']:
            vae.cuda()
            critic.cuda()
            patch_critic.cuda()

        return {
            'vae': vae,
            'critic': critic,
            'patch_critic': patch_critic,
            'critic_opt': critic_opt,
            'patch_critic_opt': patch_critic_opt,
            'enc_opt': enc_opt,
            'dec_opt': dec_opt,
        }

    def save(vae, critic, patch_critic, enc_opt, dec_opt, critic_opt, patch_critic_opt):
        torch.save({
            'vae_state_dict': vae.state_dict(),
            'critic_state_dict': critic.state_dict(),
            'patch_critic_state_dict': patch_critic.state_dict(),
            'critic_opt_state_dict': critic_opt.state_dict(),
            'patch_critic_opt_state_dict': patch_critic_opt.state_dict(),
            'enc_opt_state_dict': enc_opt.state_dict(),
            'dec_opt_state_dict': dec_opt.state_dict(),
        }, MODEL_PATH)


    def load():
        critic = Critic(**cfg['CRITIC_PARAMS'])
        patch_critic = NLayerDiscriminator(**cfg['PATCH_CRITIC_PARAMS'])
        vae = VarAutoEncoder(**cfg['VAE_PARAMS'])

        # place on device before passing parameters to optimizers!
        if cfg['cuda']:
            vae.cuda()
            critic.cuda()
            patch_critic.cuda()

        critic_opt = torch.optim.Adam(critic.parameters(), **cfg['CRITIC_OPT_PARAMS'])
        enc_opt = torch.optim.Adam(vae.encoder_params(), **cfg['ENC_OPT_PARAMS'])
        dec_opt = torch.optim.Adam(vae.decoder_params(), **cfg['DEC_OPT_PARAMS'])
        patch_critic_opt = torch.optim.Adam(patch_critic.parameters(), **cfg['PATCH_CRITIC_OPT_PARAMS'])

        data = torch.load(MODEL_PATH)

        vae.load_state_dict(data['vae_state_dict'])
        critic.load_state_dict(data['critic_state_dict'])
        critic_opt.load_state_dict(data['critic_opt_state_dict'])
        enc_opt.load_state_dict(data['enc_opt_state_dict'])
        dec_opt.load_state_dict(data['dec_opt_state_dict'])
        patch_critic_opt.load_state_dict(data['patch_critic_opt_state_dict'])

        for key, val in critic_opt.state.items():
            val['step'] = val['step'].cpu()
        for key, val in enc_opt.state.items():
            val['step'] = val['step'].cpu()
        for key, val in dec_opt.state.items():
            val['step'] = val['step'].cpu()
        for key, val in patch_critic_opt.state.items():
            val['step'] = val['step'].cpu()

        return {
            'vae': vae,
            'critic': critic,
            'patch_critic': patch_critic,
            'critic_opt': critic_opt,
            'enc_opt': enc_opt,
            'dec_opt': dec_opt,
            'patch_critic_opt': patch_critic_opt,
        }

    return new, save, load