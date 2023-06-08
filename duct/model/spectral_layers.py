"""Taken and adapted from https://github.com/wesbz/SoundStream/blob/main/net.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from duct.model.torch_modules import get_conv, get_norm, get_nonlinearity


class SpectralTransform(nn.Module):
    def __init__(
                self, 
                n_fft=1024,
                hop_length=256, 
                window_length=1024,
                window_fn=torch.hann_window,
            ):
        super(SpectralTransform, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = window_length
        self.window_fn = window_fn

    def forward(self, x):
        device = 'cuda' if x.is_cuda else 'cpu'
        window = self.window_fn(
            self.win_length, 
            device=device
        )
        x_s = torch.stft(
            x[:, 0, :],
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=False,
        )
        return x_s.permute(0, 3, 1, 2)


class SpectralResidualBlock(nn.Module):
    def __init__(
                self, 
                in_channels, 
                mid_channels, 
                out_channels, 
                s_t, 
                s_f, 
                dropout=0.1
            ):
        super().__init__()
        
        self.s_t = s_t
        self.s_f = s_f

        self.layers = nn.Sequential(
            get_conv(
                data_dim=2,
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=(3, 3),
                padding="same",
                with_weight_norm=True
            ),
            get_norm(data_dim=2, type='batch')(mid_channels),
            get_nonlinearity('ELU'),
            torch.nn.Dropout(dropout),
            get_conv(
                data_dim=2,
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=(s_f+2, s_t+2),
                stride=(s_f, s_t),
                with_weight_norm=True
            ),
            get_norm(data_dim=2, type='batch')(out_channels),
        )
        
        self.skip_connection = get_conv(
            data_dim=2,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1), 
            stride=(s_f, s_t),
            with_weight_norm=True
        )
        self.skip_norm = get_norm(data_dim=2, type='batch')(out_channels)
        self.output_activation = get_nonlinearity('ELU')
        self.output_dropout = torch.nn.Dropout(dropout)


    def forward(self, x):
        _h = F.pad(x, [self.s_t+1, 0, self.s_f+1, 0])
        _h = self.layers(_h)
        x_skip = self.skip_norm(self.skip_connection(x))
        x = self.output_activation(_h + x_skip)
        return self.output_dropout(x)


class SpectralEncoderBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.main = nn.Sequential(
            SpectralResidualBlock(
                in_channels=in_channels,
                mid_channels=mid_channels,
                out_channels=mid_channels,
                s_t=1, s_f=2),
            SpectralResidualBlock(
                in_channels=mid_channels,
                mid_channels=mid_channels,
                out_channels=out_channels,
                s_t=2, s_f=2),
        )
    
    def forward(self, x):
        return self.main(x)

    def loss(self, x, y):
        losses = []
        for layer in self.main:
            x = layer(x)
            y = layer(y)
            losses.append(F.mse_loss(x, y))
        return x, y, losses


class SpectralEncoder(nn.Module):
    def __init__(
                self, 
                nc=2,
                depth=3, 
                ndf=32
            ):
        super().__init__()
        self.layers = nn.ModuleList([
            get_conv(
                data_dim=2,
                in_channels=nc,
                out_channels=ndf,
                kernel_size=(7, 7),
            ),
            get_nonlinearity('ELU'),
        ])

        self.depth = depth
        for _ in range(depth):
            cur_ndf = ndf
            mid_ndf = 2*ndf
            ndf = 4*ndf
            spec_enc_block = SpectralEncoderBlock(
                in_channels=cur_ndf,
                mid_channels=mid_ndf,
                out_channels=ndf,
            )
            self.layers.append(spec_enc_block)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def loss(self, x, y):
        losses = []
        for layer in self.layers:
            if isinstance(layer, SpectralEncoderBlock):
                x, y, loss = layer.loss(x, y)
                losses.extend(loss)
            else:
                x = layer(x)
                y = layer(y)
        return losses
        

    