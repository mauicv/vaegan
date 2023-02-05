import pytest
import torch
from duct.model.transformer.model import MultiScaleTransformer
from duct.model.transformer.mask import resolution_mask
from duct.model.transformer.samplers import HierarchySampler


def generate_xs(emb_num=10, batch_size=10, device='cpu'):
    xs = []
    data_shapes = [
        (8, 8), 
        (16, 16), 
        (32, 32), 
        (64, 64)
    ]
    for data_shape in data_shapes:
        xs.append(torch.randint(
            0, emb_num, 
            (batch_size, *data_shape),
            dtype=torch.long,
            device=device
        ))
    return xs


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
        block_size=64)
    sampler = HierarchySampler(transformer)
    xs = generate_xs(batch_size=12)
    inds, toks = sampler.sub_sample(xs)

    token_counts = [8*8, 16*16, 32*32, 64*64]
    for ind, pos_emb in enumerate(transformer.pos_embs):
        assert pos_emb.weight.shape[0] == token_counts[ind]

    y = transformer(toks, inds=inds)

    assert y.shape == (12, 4, 64, 10)
