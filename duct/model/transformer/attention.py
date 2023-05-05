import torch
import torch.nn as nn


class AttnBlock(nn.Module):
    def __init__(self, emb_dim, block_size, n_heads=1):
        super().__init__()
        assert emb_dim % n_heads == 0
        self.n_heads = n_heads
        self.emb_dim = emb_dim
        self.block_size = block_size
        self.norm = nn.LayerNorm(emb_dim)
        self.q = torch.nn.Linear(emb_dim, emb_dim)
        self.k = torch.nn.Linear(emb_dim, emb_dim)
        self.v = torch.nn.Linear(emb_dim, emb_dim)
        self.proj_out = torch.nn.Linear(emb_dim, emb_dim)
        self.head_size = self.emb_dim // self.n_heads
        self.attn_drop = nn.Dropout(0.1)
        self.resid_drop = nn.Dropout(0.1)

    def qkv(self, x):
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

    def compute_attention(self, q, k):
        # compute attention
        w_ = q @ k.transpose(2,3) # b, nh, l, l
        w_ = w_ * (int(self.head_size)**(-0.5))
        return w_

    def attend_to_v(self, w, v):
        # attend to values
        h_ = w @ v  # b, nh, l, hs
        _, _, l, _ = h_.shape
        h_ = h_ \
            .transpose(1, 2) \
            .reshape(-1, l, self.emb_dim) \
            .contiguous() # b, l, nh*hs
        h_ = self.proj_out(h_)
        return h_

    def forward(self, x, mask=None):
        _, l, _ = x.shape
        q, k, v = self.qkv(x)
        w_ = self.compute_attention(q, k)
        if mask is not None:
            if next(self.parameters()).is_cuda: mask = mask.cuda()
            w_ = w_.masked_fill(mask, float('-inf'))

        w_ = torch.nn.functional.softmax(w_, dim=-1)
        w_ = self.attn_drop(w_)
        h_ = self.attend_to_v(w_, v)
        h_ = self.resid_drop(h_)
        return h_

    @torch.no_grad()
    def infer(self, x, prev_k=None, prev_v=None):
        q, k, v = self.qkv(x)

        if prev_k is not None and prev_v is not None:
            k = torch.cat((prev_k, k), dim=-2)  # b, nh, l, hs
            v = torch.cat((prev_v, v), dim=-2)  # b, nh, l, hs

        w_ = self.compute_attention(q, k)
        w_ = torch.nn.functional.softmax(w_, dim=-1)
        h_ = self.attend_to_v(w_, v)
        return h_, {
            'prev_k': k[:, :, -self.block_size+1:, :], 
            'prev_v': v[:, :, -self.block_size+1:, :], 
        }