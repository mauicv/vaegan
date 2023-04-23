import torch
import torch.nn as nn
from torch import nn
from duct.model.transformer.block import ConceptBlock


class ConceptEncoder(nn.Module):
		def __init__(
				self,
				dims,
				emb_dim,
				n_heads,
				stride=1,
				use_bias=True):
			super().__init__()
			assert emb_dim % n_heads == 0
			self.dims = dims
			self.layers = torch.nn.ModuleList()
			for (m, _) in dims:
				block = ConceptBlock(m, emb_dim, n_heads, use_bias=use_bias)
				self.layers.append(block)

		def forward(self, x):
			print(x.shape)
			# x = c_emb_1(x)
			return x