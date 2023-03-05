import numpy as np
import torch.nn as nn
import torch
from duct.model.encoder import Encoder
from duct.model.transformer.block import TransformerBlock


class TransformerCritic(nn.Module):
    def __init__(
          self, 
          nc, 
          ndf,  
          data_shape,
          depth=5, 
          res_blocks=tuple(0 for _ in range(5)),
          attn_blocks=tuple(0 for _ in range(5)),
          downsample_block_type='image_block',
          num_transformer_blocks=4,
        ):
        super(TransformerCritic, self).__init__()
        assert len(data_shape) in {1, 2}, "data_dim must be 1 or 2"
        self.encoder = Encoder(
            nc=nc, ndf=ndf, depth=depth,
            data_shape=data_shape,
            res_blocks=res_blocks,
            downsample_block_type=downsample_block_type,
            attn_blocks=attn_blocks
        )
        emb_dim = np.array(self.encoder.output_shape).prod()
        self.transformer_blocks = nn.Sequential(
            *(TransformerBlock(emb_dim=emb_dim) for _ in range(num_transformer_blocks))
        )
        self.cb = ComparisonBlock(emb_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.reshape(-1, np.prod(self.encoder.output_shape))
        x = self.transformer_blocks(x[None, :, :])
        return self.cb(x)

    def loss(self, x, y, layers=None):
        self.encoder.train(False) # freeze encoder while computing loss
        loss = self.encoder.loss(x, y, layer_inds=layers)
        self.encoder.train(True) # unfreeze encoder
        return loss


class ComparisonBlock(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.norm = nn.LayerNorm(emb_dim)
        self.q = torch.nn.Linear(emb_dim, emb_dim)
        self.k = torch.nn.Linear(emb_dim, emb_dim)

    def forward(self, h_):
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        w_ = q @ k.transpose(1,2) # b, b
        # w_ = w_ * (int(l)**(-0.5))
        return torch.nn.functional.softmax(w_, dim=-1)
