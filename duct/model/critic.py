import numpy as np
import torch.nn as nn
from duct.model.encoder import Encoder

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
