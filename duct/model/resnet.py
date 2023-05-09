"""Resnet implmententation taken from taming-transformers repo and adjusted.

See https://github.com/CompVis/taming-transformers
"""

import torch
from torch.nn import Module
from duct.model.activations import get_nonlinearity
from duct.model.torch_modules import get_conv, get_norm


class ResnetBlock(Module):
    def __init__(
            self, 
            in_channels, 
            out_channels=None, 
            conv_shortcut=False, 
            dropout=0.1,
            data_dim=2,
            norm_type='batch',
            activation='ELU'
        ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = get_norm(data_dim=data_dim, type=norm_type)(out_channels)
        self.conv1 = get_conv(
            data_dim,
            in_channels=in_channels, 
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.dropout1 = torch.nn.Dropout(dropout)
        self.norm2 = get_norm(data_dim=data_dim, type=norm_type)(out_channels)
        self.conv2 = get_conv(
            data_dim, 
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.dropout2 = torch.nn.Dropout(dropout)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = get_conv(
                    data_dim, 
                    in_channels=in_channels, 
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            else:
                self.nin_shortcut = get_conv(
                    data_dim,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
        self.norm3 = get_norm(data_dim=data_dim, type=norm_type)(out_channels)
        self.nonlinearity = get_nonlinearity(activation)

    def forward(self, x):
        h = x
        h = self.conv1(h)
        h = self.norm1(h)
        h = self.nonlinearity(h)
        h = self.dropout1(h)
        h = self.conv2(h)
        h = self.norm2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        x = self.norm3(x)+h
        x = self.nonlinearity(x)
        x = self.dropout2(x)
        return x