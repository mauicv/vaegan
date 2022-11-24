import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable

from utils.config import load_config


class LatentSpace(nn.Module):
    def __init__(self, input_shape, output_shape, latent_dim=2024):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.latent_dim = latent_dim
        self.flat_in_size = np.prod(self.input_shape)
        self.flat_out_size = np.prod(self.output_shape)
        self.fc1 = nn.Linear(self.flat_in_size, self.latent_dim)
        self.fc2 = nn.Linear(self.latent_dim, self.flat_out_size)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def encode(self, x):
        x = x.reshape(-1, self.flat_in_size)
        return self.leakyrelu(self.fc1(x))

    def decode(self, z):
        B, _ = z.shape
        x = self.leakyrelu(self.fc2(z))
        x = x.reshape(B, *self.output_shape)
        return [x]


class StochasticLatentSpace(nn.Module):
    def __init__(self, input_shape, output_shape, latent_dim=2024):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.latent_dim = latent_dim
        self.flat_in_size = np.prod(self.input_shape)
        self.flat_out_size = np.prod(self.output_shape)
        self.fc1 = nn.Linear(self.flat_in_size, self.latent_dim)
        self.fc2 = nn.Linear(self.flat_in_size, self.latent_dim)
        self.fc3 = nn.Linear(self.latent_dim, self.flat_out_size)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def encode(self, x):
        x = x.reshape(-1, self.flat_in_size)
        mu = self.leakyrelu(self.fc1(x))
        logvar = self.leakyrelu(self.fc2(x))
        return mu, logvar

    def decode(self, z):
        B, _ = z.shape
        x = self.leakyrelu(self.fc3(z))
        return x.reshape(B, *self.output_shape)

    def reparametrize(self, mu, logvar):
        var = torch.exp(logvar*0.5)
        normal = torch.randn(len(mu), self.latent_dim, requires_grad=True)
        if load_config()['cuda']: normal = normal.cuda()
        return normal * var + mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return [self.decode(z), mu, logvar]