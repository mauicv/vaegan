import torch
import torch.nn as nn


import torch
from torch import nn
from torch.nn import Parameter

class ConceptEncodingBlock(nn.Module):
		def __init__(self, m, emb_dim, n_heads, use_bias=True):
			super().__init__()
			assert emb_dim % n_heads == 0
			self.m = m # number of memories
			self.n_heads = n_heads
			self.emb_dim = emb_dim
			self.head_size = self.emb_dim // self.n_heads
			self.cells = Parameter(torch.zeros(self.m, self.n_heads, self.head_size))
			self.cells.data.uniform_(-1/self.m, 1/self.m)
			self.q = torch.nn.Linear(emb_dim, emb_dim)
			self.v = Parameter(torch.randn(self.m, emb_dim, emb_dim))
			if use_bias:
				self.vb = Parameter(torch.randn(self.m, emb_dim))
			self.norm = nn.LayerNorm(emb_dim)
			self.attn_drop = nn.Dropout(0.1)

		def forward(self, x):
			_, l, _ = x.shape
			h_ = self.norm(x)
			q = self.q(h_) \
				.reshape(-1, l, self.n_heads, self.head_size) \
				.transpose(1,2) # b, nh, l, hs
			v_ = torch.einsum('mwv,blv->bmlw', self.v, h_) \
					.reshape(-1, self.m, l, self.n_heads, self.head_size) # b, m, l, nh, hs
			if hasattr(self, 'vb'):
				v_ = v_ + self.vb.reshape(1, self.m, 1, self.n_heads, self.head_size) # b, m, l, nh, hs

			# compute attention
			w_ = torch.einsum('bhlv,mhv->bhml', q, self.cells) # b, nh, m, l
			w_ = w_ * (int(self.head_size)**(-0.5))
			w_ = torch.nn.functional.softmax(w_, dim=-1) 
			w_ = self.attn_drop(w_)

			# attend to values
			h_ = torch.einsum('bhml,bmlhs->bmhs', w_, v_) # b, m, nh, hs
			h_ = h_ \
				.transpose(1, 2) \
				.reshape(-1, self.m, self.emb_dim) \
				.contiguous() # b, n, nh*hs

			return h_
