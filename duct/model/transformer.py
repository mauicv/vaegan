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
    return emb[None]


def get_causal_mask(l):
    mask = torch.tril(torch.ones(l, l))
    masked_indices = mask[None, None, :l, :l] == 0
    return mask, masked_indices


def get_local_image_mask(image_size=(32, 32), patch_size=(6, 6)):
    h, w = image_size
    mask = torch.zeros((h, w, h, w))
    patch_w, patch_h = patch_size
    for i in range(h):
        for j in range(w):
            top_l_i = i - int(patch_h/2)
            top_l_j = j - int(patch_w/2)
            for ip in range(patch_h):
                for jp in range(patch_w):
                    boundary_cond_i = top_l_i + ip < h and top_l_i + ip >= 0
                    boundary_cond_j = top_l_j + jp < w and top_l_j + jp >= 0
                    boundary_conds = boundary_cond_i and boundary_cond_j 
                    if boundary_conds:
                        if ip < int(patch_h/2) or (ip == int(patch_h/2) and jp <= int(patch_w/2)):
                            mask[i, j, top_l_i + ip, top_l_j + jp] = 1

    flattend_mask = mask.reshape(h * w, h * w)
    indicies = flattend_mask[None, None, :h * w, :h * w] == 0
    return mask, indicies


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

    def forward(self, x, mask=None):
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

        if mask is not None:
            if next(self.parameters()).is_cuda: mask = mask.cuda()
            w_ = w_.masked_fill(mask, float('-inf'))

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
            n_heads=n_heads,
        )
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, x, mask=None):
        x = self.attn(x, mask=mask)
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
                n_heads=n_heads,
            )
            self.layers.append(transformer_block)

        self.linear = nn.Linear(emb_dim, emb_num)

    def forward(self, x, mask=None):
        x = self.tok_emb(x)
        _, l, emb_dim = x.shape
        pos = torch.tensor([i for i in range(l)])
        pos_emb = get_timestep_embedding(pos, emb_dim)
        if next(self.parameters()).is_cuda: pos_emb = pos_emb.cuda()
        x = x + pos_emb

        for layer in self.layers:
            x = layer(x, mask=mask)
        logits = self.linear(x)
        return logits 
