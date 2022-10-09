from pathlib import Path
import torch
import torch.nn.functional as F
from utils import KL_div_loss
import csv


class Trainer:
    def __init__(
            self, 
            critic, 
            vae, 
            crit_opt, 
            enc_opt,
            dec_opt, 
            loader,
            save_fn,
            config,
        ):
        self.logging_path = Path('./training')
        self.logging_path.mkdir(exist_ok=True)
        self.logging_path_imgs = Path('./training/imgs')
        self.logging_path_imgs.mkdir(exist_ok=True)
        self.logging_path_loss = Path('./training/losses.csv')
        self.headers = ['decoder_loss', 'gen_loss', 'critic_loss', 
                        'encoder_loss', 'kl_loss', 'recon_loss', 
                        'mu', 'mu_std', 'var']
        if not self.logging_path_loss.exists():
            with open(self.logging_path_loss, 'w') as log_f:
                writer = csv.DictWriter(log_f, fieldnames=self.headers)
                writer.writeheader()

        self.model_path = Path('./saved_models')

        self.save_fn = save_fn
        self.critic = critic
        self.vae = vae
        self.crit_opt = crit_opt
        self.enc_opt = enc_opt 
        self.dec_opt = dec_opt
        self.loader = loader
        self.config = config
        self.cuda = config['cuda']
        self.save_rate = config['save_rate']

    def logging(self, logs):
        with open(self.logging_path_loss, 'a') as log_f:
            writer = csv.DictWriter(log_f, fieldnames=self.headers)
            writer.writerow(logs)

    def train(self):
        count = 0
        for _ in range(self.config.epochs):
            for x in self.loader:
                count += 1
                if self.cuda: x = x.cuda()
            
                # Critic
                self.crit_opt.zero_grad()
                fake = self.vae(x, with_reparam=False)
                fake = fake.detach()
                fake_logits = self.critic(fake.detach())
                real_logits = self.critic(x)
                pd_loss_real = torch.mean(F.relu(1. - real_logits))
                pd_loss_fake = torch.mean(F.relu(1. + fake_logits))
                pd_loss = 0.5 * (pd_loss_real + pd_loss_fake)
                pd_loss.backward(retain_graph=False)
                self.crit_opt.step()

                # Autoencoder
                self.dec_opt.zero_grad()
                self.enc_opt.zero_grad()
                y, mu, logvar = self.vae(x)
                kl_loss = KL_div_loss(mu, logvar)
                recon_loss = self.critic.loss(x, y, layers=[1,2,3,4]).sum()
                fake_logits = self.critic(y)
                gen_loss = - fake_logits.mean()
                encoder_loss = (recon_loss + kl_loss) * self.config.r_weight
                encoder_loss.backward(retain_graph=True)
                decoder_loss = gen_loss + recon_loss * self.config.r_weight
                decoder_loss.backward(retain_graph=True)
                self.dec_opt.step()
                self.enc_opt.step()

                # Logging
                self.logging({
                    'decoder_loss': decoder_loss.cpu().detach().numpy(),
                    'gen_loss': gen_loss.cpu().detach().numpy(),
                    'critic_loss': pd_loss.cpu().detach().numpy(),
                    'encoder_loss': encoder_loss.cpu().detach().numpy(),
                    'kl_loss': kl_loss.cpu().detach().numpy(),
                    'recon_loss': recon_loss.cpu().detach().numpy(),
                    'mu': mu.detach().cpu().numpy().mean(),
                    'mu_std': mu.detach().cpu().numpy().std(),
                    'var': torch.exp(logvar*0.5).detach().cpu().numpy().mean()
                })

                # Saving
                if count % self.save_rate == 0:
                    self.save_fn(self.vae, self.critic, self.enc_opt, 
                                 self.dec_opt, self.crit_opt)
