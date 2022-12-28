"""Taken from https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/modules/transformer/mingpt.py
and adjusted."""


import math
import torch
import torch.nn as nn


def get_timestep_embedding(timesteps, embedding_dim):
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    if torch.cuda.is_available():
        emb = emb.cuda()
    return emb[None]


class AttnBlock(nn.Module):
    def __init__(self, emb_dim, n_heads=1):
        super().__init__()
        assert emb_dim % n_heads == 0
        self.n_heads = n_heads
        self.emb_dim = emb_dim
        self.norm = nn.LayerNorm(emb_dim)
        self.q = torch.nn.Linear(emb_dim, emb_dim)
        self.k = torch.nn.Linear(emb_dim, emb_dim)
        self.v = torch.nn.Linear(emb_dim, emb_dim)
        self.proj_out = torch.nn.Linear(emb_dim, emb_dim)
        self.head_size = self.emb_dim//self.n_heads

    def forward(self, x):
        _, l, _ = x.shape
        h_ = x
        h_ = self.norm(h_)
        tensor_shape = (-1, l, self.n_heads, self.head_size)
        q = self.q(h_) \
            .reshape(*tensor_shape) \
            .transpose(1,2) # b, nh, l, hs
        k = self.k(h_) \
            .reshape(*tensor_shape) \
            .transpose(1,2) # b, nh, l, hs
        v = self.v(h_) \
            .reshape(*tensor_shape)\
            .transpose(1,2) # b, nh, l, hs

        # compute attention
        w_ = q @ k.transpose(2,3) # b, nh, l, l
        w_ = w_ * (int(self.head_size)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=-1)

        # attend to values
        h_ = w_ @ v  # b, nh, l, hs
        h_ = h_ \
            .transpose(1,2) \
            .reshape(-1, l, self.emb_dim) \
            .contiguous() # b, l, nh*hs
        h_ = self.proj_out(h_)

        return x+h_


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, n_heads=1):
        super().__init__()
        self.emb_dim = emb_dim
        self.attn = AttnBlock(
            emb_dim, 
            n_heads=n_heads
        )
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, x):
        x = self.attn(x)
        x = self.mlp(x)
        return x


class Transformer(nn.Module):
    def __init__(self, emb_dim, emb_num, n_heads=1, depth=5):
        super().__init__()
        self.tok_emb = nn.Embedding(emb_num, emb_dim)
        self.emb_num = emb_num
        self.emb_dim = emb_dim
        self.depth = depth
        self.n_heads = n_heads
        self.layers = nn.ModuleList()
        for _ in range(depth):
            transformer_block = TransformerBlock(
                emb_dim, 
                n_heads=n_heads
            )
            self.layers.append(transformer_block)

        self.linear = nn.Linear(emb_dim, emb_num)

    def forward(self, x):
        x = self.tok_emb(x)
        _, l, emb_dim = x.shape
        pos = torch.tensor([i for i in range(l)])
        pos_emb = get_timestep_embedding(pos, emb_dim)
        x = x + pos_emb

        for layer in self.layers:
            x = layer(x)
        logits = self.linear(x)
        return logits 
