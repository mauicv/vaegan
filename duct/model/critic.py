import numpy as np
import torch.nn as nn
from duct.model.encoder import Encoder


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
          with_fc=True
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
        self.with_fc = with_fc
        if with_fc:
            self.fc = nn.Linear(np.prod(self.encoder.output_shape), 1)
        self.apply(weights_init)

    def forward(self, x):
        x = self.encoder(x)
        x = x.reshape(-1, np.prod(self.encoder.output_shape))
        if self.with_fc:
            return self.fc(x)
        return x

    def loss(self, x, y, layers=None):
        self.encoder.train(False) # freeze encoder while computing loss
        loss = self.encoder.loss(x, y, layer_inds=layers)
        self.encoder.train(True) # unfreeze encoder
        return loss
