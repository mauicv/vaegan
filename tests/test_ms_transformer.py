import pytest
import torch
from duct.model.transformer.model import MultiScaleTransformer
from duct.model.transformer.mask import resolution_mask


def test_resolution_mask():
    mask, _ = resolution_mask(4, 128)
    assert mask.shape == (4*128, 4*128)


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
    token_counts = [128, 512, 2048, 8192]
    inds = torch.cat([
        torch.randint(0, count, (1, ))
        for count in token_counts
    ])
    for ind, pos_emb in enumerate(transformer.pos_embs):
        assert pos_emb.weight.shape[0] == token_counts[ind]
    y = transformer(x, inds=inds)
    assert y.shape == (64, 4, 128, 10)
