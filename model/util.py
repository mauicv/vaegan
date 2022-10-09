from model.autoencoders import VarAutoEncoder
from model.critic import Critic
import torch
from pathlib import Path


def setup_models(
      CRITIC_PARAMS, 
      VAE_PARAMS, 
      CRITIC_OPT_PARAMS, 
      ENC_OPT_PARAMS, 
      DEC_OPT_PARAMS, 
      cuda):

    PATH = Path('./saved_models')
    PATH.mkdir(exist_ok=True)

    def new():
        critic = Critic(**CRITIC_PARAMS)
        vae = VarAutoEncoder(**VAE_PARAMS)
        crit_opt = torch.optim.Adam(critic.parameters(), **CRITIC_OPT_PARAMS)
        enc_opt = torch.optim.Adam(vae.get_encoder_params(), **ENC_OPT_PARAMS)
        dec_opt = torch.optim.Adam(vae.get_decoder_params(), **DEC_OPT_PARAMS)

        if cuda:
            vae.cuda()
            critic.cuda()

        return {
            'vae': vae,
            'critic': critic,
            'crit_opt': crit_opt,
            'enc_opt': enc_opt,
            'dec_opt': dec_opt,
        }

    def save(epoch, vae, critic, enc_opt, dec_opt, crit_opt):
        torch.save({
          'epoch': epoch,
          'vae_state_dict': vae.state_dict(),
          'crit_state_dict': critic.state_dict(),
          'crit_opt_state_dict': crit_opt.state_dict(),
          'enc_opt_state_dict': enc_opt.state_dict(),
          'dec_opt_state_dict': dec_opt.state_dict(),
        }, PATH)


    def load():
        critic = Critic(**CRITIC_PARAMS)
        vae = VarAutoEncoder(**VAE_PARAMS)

        # place on device before passing parameters to optimizers!
        if cuda:
            vae.cuda()
            critic.cuda()

        crit_opt = torch.optim.Adam(critic.parameters(), **CRITIC_OPT_PARAMS)
        enc_opt = torch.optim.Adam(vae.get_encoder_params(), **ENC_OPT_PARAMS)
        dec_opt = torch.optim.Adam(vae.get_decoder_params(), **DEC_OPT_PARAMS)

        data = torch.load(PATH)

        vae.load_state_dict(data['vae_state_dict']),
        critic.load_state_dict(data['crit_state_dict']),
        crit_opt.load_state_dict(data['crit_opt_state_dict']),
        enc_opt.load_state_dict(data['enc_opt_state_dict']),
        dec_opt.load_state_dict(data['dec_opt_state_dict']),

        for key, val in crit_opt.state.items():
            val['step'] = val['step'].cpu()
        for key, val in enc_opt.state.items():
          val['step'] = val['step'].cpu()
        for key, val in dec_opt.state.items():
          val['step'] = val['step'].cpu()

        return {
            'vae': vae,
            'critic': critic,
            'crit_opt': crit_opt,
            'enc_opt': enc_opt,
            'dec_opt': dec_opt,
            'epoch': data['epoch']
        }

    return new, save, load