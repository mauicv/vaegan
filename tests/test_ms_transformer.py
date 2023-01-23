import pytest
import torch
from duct.model.transformer.model import MultiScaleTransformer
from duct.model.transformer.samplers import sample_sequential
from duct.model.transformer.mask import get_local_image_mask, get_causal_mask


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
    inds = torch.randint(0, 4, (4, ))
    y = transformer(x, inds=inds)
    assert y.shape == (64, 4, 128, 10)
