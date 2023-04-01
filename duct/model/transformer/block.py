import torch.nn as nn
from duct.model.transformer.attention import AttnBlock


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, block_size, n_heads=1, attn_block=AttnBlock):
        super().__init__()
        self.emb_dim = emb_dim
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.attn = attn_block(
            emb_dim, 
            block_size,
            n_heads=n_heads,
        )
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask=mask)
        x = x + self.mlp(self.ln2(x))
        return x
