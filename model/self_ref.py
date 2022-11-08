from torch import nn
import numpy as np


class SelfRefLayer(nn.Module):
    def __init__(self, latent_dim, target_shape):
        """Self Reflection layer

        Takes a latent code and uses it to infer what features in the image to
        focus on subseqent passes.

        Args:
            latent_dim (int): Size of the autoencoder latent space
            target_shape (tuple): should be a tuple of the form: (c, h, w).
                This operations is applied to the ... of the convolutional
                layers in the autoencoder.
        """
        super(SelfRefLayer, self).__init__()
        self.latent_dim = latent_dim
        self.target_shape = target_shape
        self.target_dim = np.prod(target_shape)

        self.d1 = nn.Linear(self.latent_dim, self.latent_dim)
        self.d2 = nn.Linear(self.latent_dim, self.target_dim)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x, z):
        z_a = self.relu(self.d1(z))
        z_a = self.relu(self.d2(z_a))
        return x * z_a.reshape(x.shape)
