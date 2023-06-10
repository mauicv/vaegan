import numpy as np
import torch.nn as nn
import torch
from duct.model.encoder import Encoder
from duct.model.torch_modules import get_conv
from duct.model.spectral_layers import SpectralTransform, SpectralEncoder

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Critic(nn.Module):
    def __init__(
          self, 
          nc, 
          ndf,  
          data_shape,
          depth=5, 
          res_blocks=tuple(0 for _ in range(5)),
          attn_blocks=tuple(0 for _ in range(5)),
          ch_mult=(1, 1, 2, 2, 3),
          downsample_block_type='image_block',
          patch=False
        ):
        super(Critic, self).__init__()
        assert len(data_shape) in {1, 2}, "data_dim must be 1 or 2"
        assert len(res_blocks) == len(attn_blocks) == len(ch_mult),(
                f'len(res_blocks)={len(res_blocks)},'
                f'len(attn_blocks)={len(attn_blocks)} and '
                f'len(ch_mult)={len(ch_mult)} should all be the same length'
            )
        self.encoder = Encoder(
            nc=nc, ndf=ndf, depth=depth,
            data_shape=data_shape,
            res_blocks=res_blocks,
            ch_mult=ch_mult,
            downsample_block_type=downsample_block_type,
            attn_blocks=attn_blocks
        )
        self.patch = patch
        if not patch:
            self.fc = nn.Linear(np.prod(self.encoder.output_shape), 1)
        else:
            out_channels, *_ = self.encoder.output_shape
            self.out_conv = get_conv(
                len(data_shape), 
                in_channels=out_channels, 
                out_channels=1, 
                kernel_size=1,
                stride=1,
                padding=0
            )
        self.apply(weights_init)

    def forward(self, x):
        x = self.encoder(x)
        if self.patch:
            x = self.out_conv(x)
        else:
            x = x.reshape(-1, np.prod(self.encoder.output_shape))
            x = self.fc(x)
        return x

    def loss(self, x, y, layers=None):
        self.train(False) # freeze encoder while computing loss
        loss = self.encoder.loss(x, y, layer_inds=layers)
        self.train(True) # unfreeze encoder
        return loss


class MultiResCritic(nn.Module):
    def __init__(self, 
            nc, 
            ndf,  
            data_shape,
            depth=5, 
            res_blocks=tuple(0 for _ in range(5)),
            attn_blocks=tuple(0 for _ in range(5)),
            ch_mult=(1, 1, 2, 2, 3),
            downsample_block_type='image_block',
            patch=True,
            num_resolutions=3
        ):
        super(MultiResCritic, self).__init__()
        self.downsampler = nn.AvgPool1d(kernel_size=4, stride=2, padding=1,
                                        count_include_pad=False)
        
        self.critics = nn.ModuleList()
        for _ in range(num_resolutions):
            self.critics.append(Critic(
                nc=nc, ndf=ndf, depth=depth,
                data_shape=data_shape,
                res_blocks=res_blocks,
                ch_mult=ch_mult,
                downsample_block_type=downsample_block_type,
                attn_blocks=attn_blocks,
                patch=patch
            ))

    def forward(self, x):
        results = {}
        for critic in self.critics:
            res = tuple(x.shape[2:])
            results[res] = critic(x).squeeze()
            x = self.downsampler(x)
        return results

    def loss(self, x, y, layers=None):
        results = {}
        self.train(False) # freeze encoder while computing loss
        for critic in self.critics:
            loss = critic.encoder.loss(x, y, layer_inds=layers)
            res = tuple(x.shape[2:])
            results[res] = loss
            x = self.downsampler(x)
            y = self.downsampler(y)
        self.train(True) # unfreeze encoder
        return loss


class SpectralCritic(nn.Module):
    def __init__(self,
            nc, 
            ndf,  
            depth=5,
            patch=True,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            normalized=True
        ):
        super(SpectralCritic, self).__init__()
        self.patch = patch
        self.spectral_transform = SpectralTransform(
            n_fft=n_fft,
            hop_length=hop_length,
            window_length=win_length,
            normalized=normalized
        )
        self.spectral_encoder = SpectralEncoder(
            nc=nc, ndf=ndf, depth=depth,
        )

        self.out_conv = get_conv(
            data_dim=2,
            in_channels=self.spectral_encoder.fin_ndf,
            out_channels=1,
            kernel_size=(3, 3),
            with_weight_norm=True,
            stride=(1, 1)
        )

        self.apply(weights_init)

    def forward(self, x):
        x = self.spectral_transform(x)
        x, fmaps = self.spectral_encoder(x)
        x = self.out_conv(x)
        return x, fmaps


class MultiScaleSpectralCritic(nn.Module):
    def __init__(
            self,
            nc=1, 
            ndf=32,  
            depth=5,
            n_ffts=[1024, 2048, 512],
            hop_lengths=[256, 512, 128],
            win_lengths=[1024, 2048, 512],
            normalized=True,
            **kwargs
        ):
        super().__init__()
        assert len(n_ffts) == len(hop_lengths) == len(win_lengths)
        self.discriminators = nn.ModuleList([
            SpectralCritic(
                nc,
                n_fft=n_fft, 
                ndf=ndf,
                depth=depth,
                normalized=normalized,
                win_length=win_len, 
                hop_length=hop_len, 
                **kwargs
            )
            for n_fft, hop_len, win_len in zip(n_ffts, hop_lengths, win_lengths)
        ])
        self.num_discriminators = len(self.discriminators)

    def forward(self, x):
        logits = []
        fmaps = []
        for disc in self.discriminators:
            logit, fmap = disc(x)
            logits.append(logit)
            fmaps.append(fmap)
        return logits, fmaps

    @staticmethod
    def relative_feature_loss(real_discs_fmaps, fake_discs_fmaps):
        """see 'High Fidelity Neural Audio Compression' EnCodec paper"""
        KL = len(real_discs_fmaps) * len(real_discs_fmaps[0])
        loss_sum = 0
        for real_fmaps, fake_fmaps in zip(real_discs_fmaps, fake_discs_fmaps):
            for real_fmap, fake_fmap in zip(real_fmaps, fake_fmaps):
                l1 = torch.abs(real_fmap - fake_fmap)
                real_mean = torch.mean(real_fmap, dim=(1, 2, 3), keepdim=True)
                loss_sum = loss_sum + (l1/real_mean).sum(dim=(1, 2, 3))
        return loss_sum / KL