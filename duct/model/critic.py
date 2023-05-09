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
          downsample_block_type='image_block',
          patch=False
        ):
        super(Critic, self).__init__()
        assert len(data_shape) in {1, 2}, "data_dim must be 1 or 2"
        self.encoder = Encoder(
            nc=nc, ndf=ndf, depth=depth,
            data_shape=data_shape,
            res_blocks=res_blocks,
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


class MutliResCritic(nn.Module):
    def __init__(self, 
            nc, 
            ndf,  
            data_shape,
            depth=5, 
            res_blocks=tuple(0 for _ in range(5)),
            attn_blocks=tuple(0 for _ in range(5)),
            downsample_block_type='image_block',
            patch=True,
            num_resolutions=3
        ):
        super(MutliResCritic, self).__init__()
        self.downsampler = nn.AvgPool1d(kernel_size=4, stride=2, padding=1,
                                        count_include_pad=False)
        
        self.critics = nn.ModuleList()
        for _ in range(num_resolutions):
            self.critics.append(Critic(
                nc=nc, ndf=ndf, depth=depth,
                data_shape=data_shape,
                res_blocks=res_blocks,
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
            data_shape=(4, 8192),
            depth=5,
            patch=True,
            n_fft=1024,
            hop_length=256,
            window_length=1024,
        ):
        super(SpectralCritic, self).__init__()
        self.patch = patch
        self.spectral_transform = SpectralTransform(
            n_fft=n_fft,
            hop_length=hop_length,
            window_length=window_length
        )
        self.spectral_encoder = SpectralEncoder(
            nc=nc, ndf=ndf, depth=depth,
        )

        with torch.no_grad():
            test = torch.ones((1, *data_shape))
            test = self.spectral_transform(test)
            test = self.spectral_encoder(test)

        self.out_conv = get_conv(
            data_dim=2,
            in_channels=test.shape[1],
            out_channels=1,
            kernel_size=(test.shape[2], 1)
        )

        self.apply(weights_init)

    def forward(self, x):
        x = self.spectral_transform(x)
        x = self.spectral_encoder(x)
        x = self.out_conv(x)
        return x.squeeze()

    def loss(self, x, y, layers=None):
        self.train(False)
        x = self.spectral_transform(x)
        y = self.spectral_transform(y)
        losses = self.spectral_encoder.loss(x, y)
        self.train(True)
        if layers is not None:
            losses = [
                loss for ind, loss in enumerate(losses)
                if ind in layers
            ]
        return sum(losses) / len(losses)
