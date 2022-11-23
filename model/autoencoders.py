from model.decoder import Decoder
from model.encoder import Encoder, DownSampleBatchConv2dBlock


from itertools import chain

import torch.nn as nn
import torch
from torch.autograd import Variable


class AutoEncoder(nn.Module):
    def __init__(
          self, 
          nc, 
          ndf, 
          latent_dim, 
          depth=5, 
          img_shape=(64, 64)):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(nc=nc, ndf=ndf, depth=depth, 
                               img_shape=img_shape,
                               downsample_block_type=DownSampleBatchConv2dBlock
                               )
        self.decoder = Decoder(nc=nc, ndf=ndf, depth=depth, 
                               latent_dim=latent_dim,
                               img_shape=img_shape)
        self.fc1 = nn.Linear(
            self.encoder.encoding_output_shape, 
            latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        z = self.fc1(x)
        y = self.decoder(z)
        return y

    def get_encoder_params(self):
      return chain(
          self.encoder.parameters(), 
          self.fc1.parameters(), 
      )

    def get_decoder_params(self):
      return self.decoder.parameters()


class VarAutoEncoder(AutoEncoder):
    def __init__(
          self, 
          nc, 
          ndf, 
          latent_dim, 
          depth=5, 
          img_shape=(128, 128),
          cuda=False):
        super(VarAutoEncoder, self).__init__(nc, 
              ndf, 
              latent_dim, 
              depth=depth,
              img_shape=img_shape)
        self.latent_dim = latent_dim
        self.fc2 = nn.Linear(
            self.encoder.encoding_output_shape,
            latent_dim)
        self.cuda = cuda

    def reparametrize(self, mu, logvar):
        var = torch.exp(logvar*0.5)
        normal = Variable(
            torch.randn(len(mu), self.latent_dim),
            requires_grad=True)
        if self.cuda: normal = normal.cuda()
        return normal * var + mu

    def forward(self, x, with_reparam=True):
        x = self.encoder(x)
        if with_reparam:
          mu, logvar = self.fc1(x), self.fc2(x)
          z = self.reparametrize(mu, logvar)
          return self.decoder(z), mu, logvar
        else:
          z = self.fc1(x)
          return self.decoder(z)

    def sample(self, z=None, batch_size=64):
        if z is None:
          z = Variable(torch.randn((batch_size, self.latent_dim)), 
                      requires_grad=True)
        if self.cuda: z = z.cuda()
        return self.decoder(z)

    def get_encoder_params(self):
      return chain(
          self.encoder.parameters(), 
          self.fc1.parameters(), 
          self.fc2.parameters()
      )

    def get_decoder_params(self):
      return self.decoder.parameters()