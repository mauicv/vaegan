import numpy as np
import torch.nn as nn
from model.encoder import Encoder

class Critic(nn.Module):
    def __init__(
          self, 
          nc, 
          ndf,  
          depth=5, 
          img_shape=(128, 128),
          res_blocks=tuple(0 for _ in range(5)),):
        super(Critic, self).__init__()
        self.encoder = Encoder(
            nc=nc, ndf=ndf, depth=depth,
            img_shape=img_shape,
            res_blocks=res_blocks)
        self.fc = nn.Linear(np.prod(self.encoder.output_shape), 1)

    def forward(self, x):
        x = self.encoder(x)
        x = x.reshape(-1, np.prod(self.encoder.output_shape))
        return self.fc(x)

    def loss(self, x, y, layers=None):
        self.encoder.train(False) # freeze encoder while computing loss
        loss = self.encoder.loss(x, y, layer_inds=layers)
        self.encoder.train(True) # unfreeze encoder
        return loss
