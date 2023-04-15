import math
import torch
import torch.nn as nn
from duct.model.transformer.block import TransformerBlock
from duct.model.transformer.base_transformer import BaseTransformer
from duct.model.transformer.attention import AttnBlock
from duct.model.transformer.relative_attention import RelAttnBlock, SkewedRelAttnBlock



def get_timestep_embedding(timesteps, embedding_dim):
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb[None]


class Transformer(nn.Module, BaseTransformer):
    def __init__(
            self, 
            emb_dim, 
            emb_num, 
            block_size, 
            n_heads=1, 
            depth=5, 
            trainable_pos_embeddings=True,
        ):
        super().__init__()
        BaseTransformer.__init__(self)

        self.tok_emb = nn.Embedding(emb_num, emb_dim)
        if trainable_pos_embeddings:
            self.pos_emb = nn.Embedding(block_size, emb_dim)
        self.block_size = block_size
        self.emb_num = emb_num
        self.emb_dim = emb_dim
        self.depth = depth
        self.n_heads = n_heads
        self.layers = nn.ModuleList()
        for _ in range(depth):
            transformer_block = TransformerBlock(
                emb_dim, 
                self.block_size,
                n_heads=n_heads,
            )
            self.layers.append(transformer_block)
        self.linear = nn.Linear(emb_dim, emb_num)
        self.drop = nn.Dropout(0.1)
        self.apply(self._init_weights)

    def forward(self, x, mask=None):
        x = self.tok_emb(x)
        _, l, emb_dim = x.shape
        pos = torch.arange(0, l, dtype=torch.long, device=x.device)
        if hasattr(self, 'pos_emb'):
            pos_emb = self.pos_emb(pos.unsqueeze(0))
        else:
            pos_emb = get_timestep_embedding(pos, emb_dim)
        if next(self.parameters()).is_cuda: pos_emb = pos_emb.cuda()
        x = x + pos_emb
        x = self.drop(x)

        for layer in self.layers:
            x = layer(x, mask=mask)
        logits = self.linear(x)
        return logits


class RelEmbTransformer(nn.Module, BaseTransformer):
    def __init__(
            self, 
            emb_dim, 
            emb_num, 
            block_size, 
            n_heads=1, 
            depth=5, 
            rel_emb_type='skewed', # 'full' or 'skewed'
        ):
        super().__init__()
        BaseTransformer.__init__(self)

        self.tok_emb = nn.Embedding(emb_num, emb_dim)
        self.block_size = block_size
        self.emb_num = emb_num
        self.emb_dim = emb_dim
        self.depth = depth
        self.n_heads = n_heads
        self.layers = nn.ModuleList()
        for _ in range(depth):
            transformer_block = TransformerBlock(
                emb_dim, 
                self.block_size,
                n_heads=n_heads,
                attn_block=RelAttnBlock if rel_emb_type == 'full' \
                    else SkewedRelAttnBlock,
            )
            self.layers.append(transformer_block)
        self.linear = nn.Linear(emb_dim, emb_num)
        self.drop = nn.Dropout(0.1)
        self.apply(self._init_weights)

    def forward(self, x, mask=None):
        x = self.tok_emb(x)
        x = self.drop(x)
        for layer in self.layers:
            x = layer(x, mask=mask)
        logits = self.linear(x)
        return logits
