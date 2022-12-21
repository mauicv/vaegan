"""Resnet implmententation taken from taming-transformers repo and adjusted.

See https://github.com/CompVis/taming-transformers
"""

import torch
from torch.nn import Module
from duct.model.activations import nonlinearity
from duct.model.torch_modules import get_conv, get_batch_norm


class ResnetBlock(Module):
    def __init__(
            self, 
            in_channels, 
            out_channels=None, 
            conv_shortcut=False, 
            dropout=0,
            data_dim=2
        ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = get_batch_norm(data_dim)(in_channels)
        self.conv1 = get_conv(data_dim)(
            in_channels, out_channels, kernel_size=3,
            stride=1, padding=1)
        self.norm2 = get_batch_norm(data_dim)(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = get_conv(data_dim)(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = get_conv(data_dim)(
                    in_channels, out_channels, kernel_size=3,
                    stride=1, padding=1)
            else:
                self.nin_shortcut = get_conv(data_dim)(
                    in_channels, out_channels, kernel_size=1,
                    stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h