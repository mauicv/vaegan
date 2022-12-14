"""Taken from https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/modules/transformer/mingpt.py
and adjusted."""


import math
import torch
import torch.nn as nn
from duct.model.transformer.block import TransformerBlock
from torch.nn import functional as F


def get_timestep_embedding(timesteps, embedding_dim):
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb[None]


def top_k_logits(logits, k):
    v, _ = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[..., [-1]]] = -float('Inf')
    return out


@torch.no_grad()
def sample(model, x, temperature=1.0, top_k=50, sample=True, mask=None):
    l = x.shape[0]
    model.eval()
    block_size = model.block_size
    seq = torch.zeros(block_size, dtype=torch.long, device=x.device)
    seq[:l] = x
    seq = seq[None, :]

    for k in range(l, block_size):
        logits = model(seq, mask=mask)
        logits = logits[:, k-1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        if top_k is not None:
            logits = top_k_logits(logits, top_k)

        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        seq[0, k] = ix

    return seq[0]


class Transformer(nn.Module):
    def __init__(
            self, 
            emb_dim, 
            emb_num, 
            block_size, 
            n_heads=1, 
            depth=5, 
            trainable_pos_embeddings=True
        ):
        super().__init__()
        self.tok_emb = nn.Embedding(emb_num, emb_dim)
        if trainable_pos_embeddings:
            self.pos_emb = nn.Embedding(block_size, emb_dim)
        self.block_size = block_size
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
        self.drop = nn.Dropout(0.1)
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, mask=None):
        x = self.tok_emb(x)
        _, l, emb_dim = x.shape
        pos = torch.arange(0, l, dtype=torch.long, device=x.device)
        if hasattr(self, 'pos_emb'):
            pos_emb = self.pos_emb(pos.unsqueeze(0))
        else:
            pos_emb = get_timestep_embedding(pos, emb_dim)
        if next(self.parameters()).is_cuda: pos_emb = pos_emb.cuda()
        x = x + pos_emb
        x = self.drop(x)

        for layer in self.layers:
            x = layer(x, mask=mask)
        logits = self.linear(x)
        return logits

    def get_weight_decay_params(self):
        for _, module in self.named_modules():
            if isinstance(module, torch.nn.Linear):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        yield param

    def get_parameter_groups(self):
        decay_params = set(self.get_weight_decay_params())
        no_decay_params = [p for p in self.parameters() if p not in decay_params]
        groups = [
            {
                'params': list(decay_params),
                'weight_decay': 0.01,
            },
            {
                'params': no_decay_params,
                'weight_decay': 0.0,
            },
        ]
        return groups
