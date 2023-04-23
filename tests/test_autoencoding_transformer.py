import pytest
import torch
from duct.model.transformer.encoding_block import ConceptEncodingBlock
from duct.model.transformer.autoencoder_transformer import AutoEncodingTransformer
from duct.model.transformer.concept_encoder import ConceptEncoder


@pytest.mark.parametrize("n_heads", [1, 2, 4, 8])
def test_concept_encoding_block(n_heads):
    attn_block = ConceptEncodingBlock(m=10, emb_dim=64, n_heads=n_heads)
    x = torch.randn(64, 20, 64)
    y = attn_block(x)
    assert y.shape == (64, 10, 64)


@pytest.mark.parametrize("n_heads", [1, 2, 4, 8])
def test_autoencoding_transformer(n_heads):
    transformer = AutoEncodingTransformer(
        n_heads=n_heads,
        emb_dim=64,
        emb_num=10,
        n_concepts=10,
        encoder_depth=5,
        encoder_width=40,
        decoder_depth=5,
        decoder_width=20,
    )
    x = torch.randint(0, 10, (64, 20))
    y = torch.randint(0, 10, (64, 40))
    x_logits = transformer(x, y)
    assert x_logits.shape == (64, 20, 10)  # (batch_size, block_size, emb_num)


@pytest.mark.parametrize("n_heads", [1, 2, 4, 8])
def test_autoencoding_transformer_2(n_heads):
    transformer = AutoEncodingTransformer(
        n_heads=n_heads,
        emb_dim=64,
        emb_num=10,
        n_concepts=10,
        encoder_depth=5,
        encoder_width=40,
        decoder_depth=5,
        decoder_width=20,
    )
    x = torch.randint(0, 10, (64, 20))
    y = torch.randint(0, 10, (64, 120))
    x_logits = transformer(x, y)
    assert x_logits.shape == (64, 20, 10)  # (batch_size, block_size, emb_num)


@pytest.mark.parametrize("n_heads", [1, 2, 4, 8])
def test_conceptformer(n_heads):
    concept_encoder = ConceptEncoder(
        dims=((256, 64), (64, 32)), 
        emb_dim=64, 
        n_heads=n_heads, 
        stride=128,
        use_bias=True
    )
    x = torch.randn(64, 512)
    concept_encoder(x)