import pytest
import torch
from duct.model.transformer.model import Transformer, sample
from duct.model.transformer.block import TransformerBlock, AttnBlock
from duct.model.transformer.mask import get_local_image_mask, get_causal_mask


@pytest.mark.parametrize("n_heads", [1, 2, 4, 8])
def test_attn_block(n_heads):
    attn_block = AttnBlock(emb_dim=64, n_heads=n_heads)
    x = torch.randn(64, 128, 64)
    y = attn_block(x)
    assert y.shape == x.shape


@pytest.mark.parametrize("n_heads", [1, 2, 4, 8])
def test_transformer_block(n_heads):
    transformer = TransformerBlock(n_heads=n_heads, emb_dim=64)
    x = torch.randn(64, 128, 64)
    y = transformer(x)
    assert y.shape == x.shape


@pytest.mark.parametrize("n_heads", [1, 2, 4, 8])
def test_transformer(n_heads):
    transformer = Transformer(n_heads=n_heads, emb_dim=256, emb_num=10, depth=5)
    x = torch.randint(0, 10, (64, 128))
    y = transformer(x)
    assert y.shape == (64, 128, 10)


@pytest.mark.parametrize("n_heads", [1, 2, 4, 8])
def test_transformer_img_mask(n_heads):
    transformer = Transformer(n_heads=n_heads, emb_dim=256, emb_num=10, depth=5)
    x = torch.randint(0, 10, (32, 8*8))
    _, mask = get_local_image_mask((8,8), (4, 4))
    mask = None
    y = transformer(x, mask=mask)
    assert y.shape == (32, 8*8, 10)
    assert torch.nan not in y


@pytest.mark.parametrize("n_heads", [1, 2, 4, 8])
def test_transformer_aud_mask(n_heads):
    transformer = Transformer(n_heads=n_heads, emb_dim=256, emb_num=10, depth=5)
    x = torch.randint(0, 10, (64, 128))
    _, mask = get_causal_mask(x.shape[1])
    y = transformer(x, mask=mask)
    assert y.shape == (64, 128, 10)
    assert torch.nan not in y


def test_sample_trasnformer():
    transformer = Transformer(n_heads=1, emb_dim=256, emb_num=10, depth=5, block_size=8*8)
    x = torch.randint(0, 10, (8, ))
    _, mask = get_local_image_mask((8,8), (4, 4))
    y = sample(transformer, x, sample=False, mask=mask)
    assert torch.all(y[:x.shape[0]] == x)
    assert torch.any(y[x.shape[0]:] > 0)
    y = sample(transformer, x, sample=True, mask=mask)
    assert torch.all(y[:x.shape[0]] == x)
    assert torch.any(y[x.shape[0]:] > 0)
