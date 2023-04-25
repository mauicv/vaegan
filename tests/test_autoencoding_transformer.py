import pytest
import torch
from duct.model.transformer.encoding_block import ConceptEncodingBlock
from duct.model.transformer.autoencoder_transformer import AutoEncodingTransformer
from duct.model.transformer.concept_encoder import ConceptEncoder, ConceptConv


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
    layers = [
        {
            'nodes': 64, 
            'width': 256, 
            'stride': 128
        },
        {
            'nodes': 32, 
            'width': 64, 
            'stride': 128
        }
    ]
    concept_encoder = ConceptEncoder(
        emb_dim=64,
        emb_num=10,
        n_heads=n_heads, 
        use_bias=True,
        layers=layers, 
    )
    x = torch.randint(0, 10, (64, 512))
    y = concept_encoder(x)
    assert y.shape == (64, 64, 10)


def test_ConceptConv():
    concept_conv = ConceptConv(
        nodes=8,
        width=64,
        blocks=1,
        stride=32,
        emb_dim=64,
        n_heads=1
    )
    x = torch.randn(12, 128, 64) # (batch_size, block_size, emb_dim)
    y = concept_conv(x)
    assert y.shape == (12, 24, 64)
