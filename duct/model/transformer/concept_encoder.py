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
from duct.model.transformer.base_transformer import BaseTransformer


class ConceptEncoder(nn.Module, BaseTransformer):
	def __init__(
			self,
			layers, # [{'nodes': ..., 'width': ..., 'stride': ...},...]
			emb_dim,
			n_heads,
			emb_num,
			use_bias=True):
		super().__init__()
		assert emb_dim % n_heads == 0
		self.emb_dim = emb_dim
		self.emb_num = emb_num
		self.layers_params = layers
		self.drop = nn.Dropout(0.1)
		self.tok_emb = nn.Embedding(emb_num, emb_dim)
		self.linear = torch.nn.Linear(emb_dim, emb_num)
		self.layers = torch.nn.ModuleList()
		for layer in layers:
			block = ConceptConv(
				**layer,
				emb_dim=emb_dim,
				n_heads=n_heads,
				use_bias=use_bias
			)
			self.layers.append(block)

	def forward(self, x):
		x = self.tok_emb(x)
		x = self.drop(x)
		for layer in self.layers:
			x = layer(x)
		logits = self.linear(x)
		return logits


class ConceptConv(nn.Module):
	def __init__(
				self,
				nodes,
				width,
				stride,
				emb_dim,
				n_heads,
				blocks=1,
				attn_type='attn',
				use_bias=True,
			) -> None:
		super().__init__()
		self.emb_dim = emb_dim
		self.nodes = nodes
		self.width = width
		attn_block = {
			'attn': AttnBlock,
			'rel_attn': RelAttnBlock,
			'skewed_rel_attn': SkewedRelAttnBlock
		}[attn_type]
		self.stride = stride
		self.concept_block = ConceptBlock(
			use_bias=use_bias,
			n_heads=n_heads,
			emb_dim=emb_dim,
			n_concepts=nodes,
		)
		self.blocks = nn.ModuleList()
		for _ in range(blocks):
			block = TransformerBlock(
				attn_block=attn_block,
				block_size=width,
				emb_dim=emb_dim, 
				n_heads=n_heads, 
			)
			self.blocks.append(block)

		self.unfold = nn.Unfold(kernel_size=(1, width), stride=1)

	def forward(self, x):
		b, _, _ = x.shape
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
