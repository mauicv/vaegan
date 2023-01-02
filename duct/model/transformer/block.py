import math
import torch
import torch.nn as nn


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
        self.head_size = self.emb_dim // self.n_heads
        self.attn_drop = nn.Dropout(0.1)
        self.resid_drop = nn.Dropout(0.1)

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
        w_ = self.attn_drop(w_)

        # attend to values
        h_ = w_ @ v  # b, nh, l, hs
        h_ = h_ \
            .transpose(1, 2) \
            .reshape(-1, l, self.emb_dim) \
            .contiguous() # b, l, nh*hs
        h_ = self.resid_drop(self.proj_out(h_))

        return h_


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, n_heads=1):
        super().__init__()
        self.emb_dim = emb_dim
        self.attn = AttnBlock(
            emb_dim, 
            n_heads=n_heads,
        )
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x, mask=None):
        x = x + self.attn(x, mask=mask)
        x = x + self.mlp(x)
        return x
