"""Relative attention block for transformer.

This is a slightly modified version of the implementation from: 
https://github.com/chathasphere/pno-ai/blob/master/model/attention.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelAttnBlock(nn.Module):
    def __init__(self, emb_dim, block_size, n_heads=1):
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
        self.Er = nn.Parameter(torch.randn(
            self.n_heads, 
            block_size,
            self.head_size
        ))
        self.block_size = block_size

    def forward(self, x, mask=None):
        if mask is not None:
            if next(self.parameters()).is_cuda: mask = mask.cuda()

        _, l, _ = x.shape
        embedding_start = self.block_size - l

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

        Er = self.Er[:, embedding_start:, :].unsqueeze(0)
        QEr = torch.matmul(q, Er.transpose(-1, -2))
        SRel = self._skew(QEr)

        # compute attention
        w_ = q @ k.transpose(2,3) # b, nh, l, l
        w_ = w_ * (int(self.head_size)**(-0.5))

        if mask is not None:
            w_ = w_.masked_fill(mask, float('-inf'))
            SRel = SRel.masked_fill(mask, float('-inf'))

        w_ = w_ + SRel

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

    def _skew(self, qe):
        padded_qe = F.pad(qe, [1,0])
        s = padded_qe.shape
        padded_qe = padded_qe.view(s[0], s[1], s[3], s[2])
        return padded_qe[:,:,1:,:]
