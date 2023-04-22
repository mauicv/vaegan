import torch
import torch.nn as nn
from duct.model.transformer.block import TransformerBlock, ConceptBlock, DecoderTransformerBlock
from duct.model.transformer.base_transformer import BaseTransformer
from duct.model.transformer.relative_attention import SkewedRelAttnBlock, RelAttnBlock


class ARPTransformer(nn.Module, BaseTransformer):
    def __init__(
            self, 
            emb_dim, 
            emb_num, 
            n_heads=1, 
            encoder_depth=5,
            encoder_width=1024,
            decoder_depth=5,
            decoder_width=512,
            n_concepts=256,
            attention_type='skewed'
        ):
        super().__init__()
        if attention_type == 'skewed':
            attn_block = SkewedRelAttnBlock
        else:
            attn_block = RelAttnBlock

        self.emb_num = emb_num
        self.emb_dim = emb_dim
        self.encoder_width = encoder_width
        self.encoder_depth = encoder_depth
        self.decoder_width = decoder_width
        self.decoder_depth = decoder_depth
        self.n_heads = n_heads
        self.n_concepts = n_concepts

        self.tok_emb = nn.Embedding(emb_num, emb_dim)

        self.encoder = Encoder(
            emb_dim,
            emb_num,
            encoder_width,
            n_heads=n_heads,
            depth=encoder_depth,
            n_concepts=n_concepts,
            attn_block=attn_block,
        )

        self.decoder = Decoder(
            emb_dim,
            emb_num,
            decoder_width,
            n_heads=n_heads,
            depth=decoder_depth,
            attn_block=attn_block,
        )
        
        self.drop = nn.Dropout(0.1)
        self.linear = nn.Linear(emb_dim, emb_num)
        self.apply(self._init_weights)

    def _preprocess(self, x):
        x = self.tok_emb(x)
        _, l, _ = x.shape
        x = self.drop(x)
        return x

    def forward(self, x, y, mask=None):
        x = self._preprocess(x)
        b, _ = y.shape
        y = y.reshape(-1, self.encoder_width)
        y = self._preprocess(y)
        y = self.encoder(y) # (b*e, n, e_w)
        y = y.reshape(b, self.encoder.n_concepts, self.emb_dim, -1).sum(-1)
        x = self.decoder(x, enc=y, mask=mask)
        logits = self.linear(x)
        return logits


class AutoEncodingTransformer(nn.Module, BaseTransformer):
    def __init__(
            self, 
            emb_dim, 
            emb_num, 
            n_heads=1, 
            encoder_depth=5,
            encoder_width=1024,
            decoder_depth=5,
            decoder_width=512,
            n_concepts=256
        ):
        super().__init__()

        self.emb_num = emb_num
        self.emb_dim = emb_dim
        self.encoder_width = encoder_width
        self.encoder_depth = encoder_depth
        self.decoder_width = decoder_width
        self.decoder_depth = decoder_depth
        self.n_heads = n_heads
        self.n_concepts = n_concepts

        self.tok_emb = nn.Embedding(emb_num, emb_dim)
        self.pos_emb = nn.Embedding(
            max(self.encoder_width, self.decoder_width), 
            emb_dim
        )

        self.encoder = Encoder(
            emb_dim,
            emb_num,
            encoder_width,
            n_heads=n_heads,
            depth=encoder_depth,
            n_concepts=n_concepts
        )

        self.decoder = Decoder(
            emb_dim,
            emb_num,
            decoder_width,
            n_heads=n_heads,
            depth=decoder_depth,
        )
        
        self.drop = nn.Dropout(0.1)
        self.linear = nn.Linear(emb_dim, emb_num)
        self.apply(self._init_weights)

    def _preprocess(self, x):
        x = self.tok_emb(x)
        _, l, _ = x.shape
        pos = torch.arange(0, l, dtype=torch.long, device=x.device)
        pos_emb = self.pos_emb(pos.unsqueeze(0))
        if next(self.parameters()).is_cuda: pos_emb = pos_emb.cuda()
        x = x + pos_emb
        x = self.drop(x)
        return x

    def forward(self, x, y, mask=None):
        x = self._preprocess(x)
        b, _ = y.shape
        y = y.reshape(-1, self.encoder_width)
        y = self._preprocess(y)
        y = self.encoder(y) # (b*e, n, e_w)
        y = y.reshape(b, self.encoder.n_concepts, self.emb_dim, -1).sum(-1)
        x = self.decoder(x, enc=y, mask=mask)
        logits = self.linear(x)
        return logits


class Decoder(nn.Module, BaseTransformer):
    def __init__(
            self, 
            emb_dim, 
            emb_num, 
            block_size, 
            n_heads=1, 
            depth=5,
            attn_block=SkewedRelAttnBlock
        ):
        super().__init__()

        self.block_size = block_size
        self.emb_num = emb_num
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.depth = depth

        self.layers = nn.ModuleList()
        for _ in range(depth):
            transformer_block = DecoderTransformerBlock(
                emb_dim, 
                self.block_size,
                n_heads=n_heads,
                attn_block=attn_block
            )
            self.layers.append(transformer_block)
        self.apply(self._init_weights)

    def forward(self, x, enc, mask=None):
        for layer in self.layers:
            x = layer(x, enc=enc, mask=mask)
        return x


class Encoder(nn.Module, BaseTransformer):
    def __init__(
            self, 
            emb_dim, 
            emb_num, 
            block_size, 
            n_heads=1, 
            depth=5,
            n_concepts=256,
            attn_block=SkewedRelAttnBlock
        ):
        super().__init__()

        self.n_concepts = n_concepts
        self.block_size = block_size
        self.emb_num = emb_num
        self.emb_dim = emb_dim
        self.encoder_depth = depth
        self.n_heads = n_heads
        self.layers = nn.ModuleList()
        for _ in range(depth):
            transformer_block = TransformerBlock(
                emb_dim, 
                self.block_size,
                n_heads=n_heads,
                attn_block=attn_block
            )
            self.layers.append(transformer_block)

        self.concept_block = ConceptBlock(
            emb_dim,
            n_concepts, 
            n_heads=n_heads
        )

        self.apply(self._init_weights)

    def forward(self, seq):
        for layer in self.layers:
            seq = layer(seq)
        seq = self.concept_block(seq)
        return seq
