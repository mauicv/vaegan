"""Taken from https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/modules/transformer/mingpt.py
and adjusted."""


import math
import torch
import torch.nn as nn
from duct.model.transformer.block import TransformerBlock
from duct.model.transformer.base_transformer import BaseTransformer


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
            trainable_pos_embeddings=True
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


class MultiScaleTransformer(nn.Module, BaseTransformer):
    def __init__(
            self, 
            emb_dim, 
            emb_num, 
            block_size, 
            num_scales,
            n_heads=1, 
            depth=5, 
        ):
        super().__init__()
        BaseTransformer.__init__(self)
        self.num_scales = num_scales
        self.block_size = block_size
        self.emb_num = emb_num
        self.emb_dim = emb_dim
        self.depth = depth
        self.n_heads = n_heads

        self.tok_emb = nn.Embedding(emb_num, emb_dim)
        self.pos_emb = nn.Embedding(block_size, emb_dim)
        self.ind_emb = nn.Embedding(num_scales, emb_dim)
        self.scale_emb = nn.Embedding(num_scales, emb_dim)

        self.layers = nn.ModuleList()
        for _ in range(depth):
            transformer_block = TransformerBlock(
                emb_dim, 
                n_heads=n_heads,
            )
            self.layers.append(transformer_block)
        self.linear = nn.Linear(emb_dim, emb_num)
        self.drop = nn.Dropout(0.1)
        self.apply(self._init_weights)

    def _preprocess_input(self, x):
        b, l, s, t = x.shape
        return x.reshape(b, l*s, t)

    def forward(self, x, inds):
        _, s, l = x.shape
        assert len(inds) == s
        x = self.tok_emb(x)

        pos = torch.arange(0, l, dtype=torch.long, device=x.device)
        pos_emb = self.pos_emb(pos)
        if next(self.parameters()).is_cuda: pos_emb = pos_emb.cuda()

        scale = torch.arange(0, s, dtype=torch.long, device=x.device)
        scale_emb = self.scale_emb(scale)
        if next(self.parameters()).is_cuda: scale_emb = scale_emb.cuda()

        ind_emb = self.ind_emb(inds)
        if next(self.parameters()).is_cuda: ind_emb = ind_emb.cuda()

        x = x + pos_emb[None, None, :, :] \
            + scale_emb[None, :, None, :] \
            + ind_emb[None, :, None, :]
        
        x = self._preprocess_input(x)
        x = self.drop(x)

        for layer in self.layers:
            x = layer(x)
        logits = self.linear(x)

        return self._postprocess_input(logits)

    def _postprocess_input(self, x):
        return x.reshape(-1, self.num_scales, self.block_size, self.emb_num)
