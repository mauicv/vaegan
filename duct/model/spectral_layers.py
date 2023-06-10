"""Taken and adapted from https://github.com/wesbz/SoundStream/blob/main/net.py"""

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from duct.model.torch_modules import get_conv, get_norm, get_nonlinearity
from einops import rearrange


def get_2d_padding(kernel_size, dilation = (1, 1)):
    return (((kernel_size[0] - 1) * dilation[0]) // 2, ((kernel_size[1] - 1) * dilation[1]) // 2)


class SpectralTransform(nn.Module):
    def __init__(
                self, 
                n_fft=1024,
                hop_length=256, 
                window_length=1024,
                window_fn=torch.hann_window,
                normalized=True
            ):
        super(SpectralTransform, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = window_length
        self.window_fn = window_fn
        self.normalized = normalized

        self.transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window_fn=torch.hann_window,
            normalized=self.normalized,
            center=False,
            pad_mode=None,
            power=None
        )

    def forward(self, x):
        x_s = self.transform(x)
        z = torch.cat([x_s.real, x_s.imag], dim=1)
        z = rearrange(z, 'b c w t -> b c t w')
        return z


class SpectralEncoder(nn.Module):
    def __init__(
                self, 
                nc=1,
                depth=3, 
                ndf=32,
                ch_mult=(2, 2, 2),
                dilations=(1, 2, 4),
                kernel_size=(3, 8),
                stride=(1, 2),
            ):
        super().__init__()
        self.non_linearity = get_nonlinearity('leakyrelu')
        self.layers = nn.ModuleList([
            get_conv(
                data_dim=2,
                in_channels=2*nc,
                out_channels=ndf,
                kernel_size=(3, 8),
                with_weight_norm=True,
            ),
        ])

        self.depth = depth
        for ch_factor, dilation in zip(ch_mult, dilations):
            cur_ndf = ndf
            ndf = ch_factor*ndf
            padding = get_2d_padding(kernel_size, (dilation, 1))
            block = [
                get_conv(
                    data_dim=2,
                    in_channels=cur_ndf,
                    out_channels=ndf,
                    kernel_size=kernel_size,
                    with_weight_norm=True,
                    dilation=(dilation, 1),
                    stride=stride,
                    padding=padding
                ),
            ]
            self.layers.extend(block)
        self.fin_ndf = ndf

    def forward(self, x):
        # x: (B, C, T, F)
        fmaps = []
        for layer in self.layers:
            x = layer(x)
            x = self.non_linearity(x)
            fmaps.append(x)
        return x, fmaps