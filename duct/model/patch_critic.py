"""
Implementation adapted from 
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
"""

import functools
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class PatchCritic2D(nn.Module):
    def __init__(self, nc=3, ndf=64, n_layers=3):
        super(PatchCritic2D, self).__init__()
        norm_layer = nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(nc, ndf, kernel_size=kw, stride=2, padding=padw), 
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev, ndf * nf_mult, 
                    kernel_size=kw, stride=2, padding=padw, 
                    bias=False
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev, ndf * nf_mult, 
                kernel_size=kw, stride=1, padding=padw,
                bias=False
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(
                ndf * nf_mult, 1, kernel_size=kw,
                stride=1, padding=padw
            )]
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        return self.main(input)


class PatchCritic1D(nn.Module):
    def __init__(self, nc=2, ndf=64, n_layers=3):
        super(PatchCritic1D, self).__init__()

        kw = 12
        padw = 5
        sequence = [
            nn.Conv1d(nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv1d(
                    ndf * nf_mult_prev, 
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=False
                ),
                nn.BatchNorm1d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv1d(
                ndf * nf_mult_prev, 
                ndf * nf_mult, 
                kernel_size=kw, 
                stride=1, 
                padding=padw,
                bias=False
            ),
            nn.BatchNorm1d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv1d(
                ndf * nf_mult, 1, 
                kernel_size=kw, 
                stride=1,
                padding=padw
            )]
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        return self.main(input)