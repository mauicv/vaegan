import imp
import torch.nn as nn
from model.blocks import Encoder

class Critic(nn.Module):
    def __init__(
          self, 
          nc, 
          ndf,  
          depth=5, 
          img_shape=(128, 128)):
        super(Critic, self).__init__()
        self.encoder = Encoder(
            nc=nc, ndf=ndf, depth=depth,
            img_shape=img_shape, 
            norm_type='instance')
        self.fc = nn.Linear(self.encoder.encoding_output_shape, 1)

    def forward(self, x):
        x = self.encoder(x)
        return self.fc(x)

    def loss(self, x, y, layers=None):
        return self.encoder.loss(x, y, layer_inds=layers)

    def clip(self):
        for p in self.parameters():
            p.data.clamp_(-0.01, 0.01)