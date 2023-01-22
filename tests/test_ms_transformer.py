import pytest
import torch
from duct.model.transformer.model import MultiScaleTransformer
from duct.model.transformer.samplers import sample_sequential
from duct.model.transformer.mask import get_local_image_mask, get_causal_mask


@pytest.mark.parametrize("n_heads", [4, 8])
def test_ms_transformer_preprocessing(n_heads):
    transformer = MultiScaleTransformer(
        n_heads=n_heads, 
        emb_dim=256, 
        emb_num=10, 
        depth=5, 
        num_scales=4,
        block_size=128)
    x = torch.randint(0, 10, (64, 4, 128))
    assert x.max() == 9
    assert x.min() == 0
    x = transformer._preprocess_input(x)
    assert x.max() == 39
    assert x.min() == 0
    assert x.shape == (64, 4*128)


@pytest.mark.parametrize("n_heads", [4])
def test_ms_transformer_forward(n_heads):
    transformer = MultiScaleTransformer(
        n_heads=n_heads, 
        emb_dim=256, 
        emb_num=10, 
        depth=5, 
        num_scales=4,
        block_size=128)
    x = torch.randint(0, 10, (64, 4, 128))
    y = transformer(x)
    assert y.shape == (64, 4, 128, 10)
