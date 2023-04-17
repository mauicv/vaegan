import pytest
import torch
from duct.model.transformer.encoding_block import ConceptEncodingBlock
from duct.model.transformer.autoencoder_transformer import AutoEncodingTransformer


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
        encoder_depth=5,
        decoder_depth=5,
        block_size=20,
        n_concepts=10)
    x = torch.randint(0, 10, (64, 20))
    y = transformer(x)
    assert y.shape == (64, 20, 10)  # (batch_size, block_size, emb_num)
