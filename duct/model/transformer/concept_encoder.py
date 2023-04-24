"""Resources:

1. https://pytorch.org/docs/stable/generated/torch.as_strided.html
2. https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html

"""

import torch
import torch.nn as nn
from torch import nn
from duct.model.transformer.block import ConceptBlock, TransformerBlock
from duct.model.transformer.attention import AttnBlock
from duct.model.transformer.relative_attention import RelAttnBlock, SkewedRelAttnBlock 


class ConceptEncoder(nn.Module):
	def __init__(
			self,
			layers, # [{'nodes': ..., 'width': ..., 'blocks': ..., 'block_type': ''},...]
			emb_dim,
			n_heads,
			emb_num,
			stride=1,
			use_bias=True):
		super().__init__()
		assert emb_dim % n_heads == 0
		self.emb_dim = emb_dim
		self.layers = layers
		self.widths = [w for (_, w) in layers]
		self.tok_emb = nn.Embedding(emb_num, emb_dim)
		self.layers = torch.nn.ModuleList()
		for (m, _) in layers:
			block = ConceptBlock(m, emb_dim, n_heads, use_bias=use_bias)
			self.layers.append(block)


	def forward(self, x):
		b, _ = x.shape
		x = self.tok_emb(x)
		print(x.shape)
		for block, width in zip(self.layers, self.widths):
			x = x.reshape(-1, width, self.emb_dim)
			print(width, x.shape)
			x = block(x).reshape(b, -1, self.emb_dim)
			print(width, x.shape)
		# x = c_emb_1(x)
		return x


class ConceptConv(nn.Module):
	def __init__(self, nodes, width, blocks, stride, emb_dim, n_heads, attn_type='attn') -> None:
		super().__init__()
		self.emb_dim = emb_dim
		self.nodes = nodes
		self.width = width
		self.blocks = blocks
		attn_block = {
			'attn': AttnBlock,
			'rel_attn': RelAttnBlock,
			'skewed_rel_attn': SkewedRelAttnBlock
		}[attn_type]
		self.stride = stride
		self.concept_block = ConceptBlock(emb_dim, nodes, n_heads=n_heads)
		self.blocks = nn.ModuleList()
		for _ in range(blocks):
			block = TransformerBlock(
				emb_dim=emb_dim, 
				n_heads=n_heads, 
				attn_block=attn_block,
				block_size=width,
			)
			self.blocks.append(block)

		self.unfold = nn.Unfold(kernel_size=(1, width), stride=1)

	def forward(self, x):
		b, l, c = x.shape
		x = x.permute(0, 2, 1) # (b, c, l)
		x_unfolded = x \
			.unfold(-1, self.width, self.stride) \
			.permute(0, 2, 1, 3) \
			.reshape(-1, self.width, self.emb_dim)
		for block in self.blocks:
			x_unfolded = block(x_unfolded)
		x_unfolded = self.concept_block(x_unfolded)
		x = x_unfolded.reshape(b, -1, self.emb_dim)
		return x