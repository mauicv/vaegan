"""Relative attention block for transformer.

This is a slightly modified version of the implementation from: 
https://github.com/chathasphere/pno-ai/blob/master/model/attention.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SkewedRelAttnBlock(nn.Module):
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

    def _skew(self, qe):
        padded_qe = F.pad(qe, [1,0])
        s = padded_qe.shape
        padded_qe = padded_qe.view(s[0], s[1], s[3], s[2])
        return padded_qe[:,:,1:,:]

    def forward(self, x, mask=None):
        if mask is not None:
            mask = mask.to(next(self.parameters()).device)

        _, l, _ = x.shape
        q, k, v = self._qkv(x)

        embedding_start = self.block_size - l
        Er = self.Er[:, embedding_start:, :].unsqueeze(0)
        QEr = torch.matmul(q, Er.transpose(-1, -2))
        SRel = self._skew(QEr)

        w_ = self._compute_attention(q, k)

        if mask is not None:
            w_ = w_.masked_fill(mask, float('-inf'))
            SRel = SRel.masked_fill(mask, float('-inf'))

        w_ = w_ + SRel
        w_ = torch.nn.functional.softmax(w_, dim=-1)
        w_ = self.attn_drop(w_)
        h_ = self._attend_to_v(w_, v)
        h_ = self.resid_drop(h_)
        return h_

    def _qkv(self, x):
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
        return q, k, v

    def _compute_attention(self, q, k):
        # compute attention
        w_ = q @ k.transpose(2,3) # b, nh, l, l
        w_ = w_ * (int(self.head_size)**(-0.5))
        return w_

    def _attend_to_v(self, w, v):
        # attend to values
        print('w.shape', w.shape)
        print('v.shape', v.shape)
        h_ = w @ v  # b, nh, l, hs
        _, _, l, _ = h_.shape
        h_ = h_ \
            .transpose(1, 2) \
            .reshape(-1, l, self.emb_dim) \
            .contiguous() # b, l, nh*hs
        h_ = self.proj_out(h_)
        return h_

    @torch.no_grad()
    def infer(self, x, prev_k=None, prev_v=None, prev_q=None):
        q, k, v = self._qkv(x)
        if (prev_k, prev_v, prev_q) != (None, None, None):
            k = torch.cat((prev_k, k), dim=-2)  # b, nh, l, hs
            v = torch.cat((prev_v, v), dim=-2)  # b, nh, l, hs
            q = torch.cat((prev_q, q), dim=-2)  # b, nh, l, hs
        l = prev_q.shape[2] + 1 if prev_q is not None else 1
        embedding_start = self.block_size - l
        Er = self.Er[:, embedding_start:, :].unsqueeze(0)
        QEr = torch \
            .einsum('bnlh,rnlh->bnl', q, Er) \
            .unsqueeze(-2)
        w_ = self._compute_attention(q[:, :, -1:, :], k)
        w_ = w_ + QEr
        w_ = torch.nn.functional.softmax(w_, dim=-1)
        h_ = self._attend_to_v(w_, v)
        return h_, {'prev_k': k, 'prev_v': v, 'prev_q': q}

def generate_relative_positions(L):
    positions = torch.arange(L).unsqueeze(0) - torch.arange(L).unsqueeze(1)
    return positions


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
            mask = mask.to(next(self.parameters()).device)

        _, l, _ = x.shape
        rel_pos = generate_relative_positions(l)
        Er = self.Er[:, rel_pos].unsqueeze(0)

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
            .reshape(*tensor_shape) \
            .transpose(1,2) # b, nh, l, hs

        # compute attention
        w_ = q @ k.transpose(2,3) # b, nh, l, l
        w_ = w_ * (int(self.head_size)**(-0.5))

        QEr = torch.einsum('bnlh,rnlkh->bnlk', q, Er)  # b, nh, l, l

        if mask is not None:
            w_ = w_.masked_fill(mask, float('-inf'))
            QEr = QEr.masked_fill(mask, float('-inf'))

        w_ = w_ + QEr

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
