import pytest
import torch
from duct.model.transformer.model import MultiScaleTransformer
from duct.model.transformer.mask import get_resolution_mask, get_causal_mask
from duct.model.samplers import HierarchicalSequentialSampler


def generate_xs(emb_num=10, batch_size=10, device='cpu'):
    xs = []
    for data_shape in [8, 16, 32, 64]:
        xs.append(torch.randint(
            0, emb_num, 
            (batch_size, data_shape),
            dtype=torch.long,
            device=device
        ))
    return xs


def test_resolution_mask():
    mask, _ = get_resolution_mask(4, 128)
    assert mask.shape == (4*128, 4*128)


@pytest.mark.parametrize("n_heads", [1, 4])
@pytest.mark.parametrize("mask_type", ['causal', 'none'])
def test_ms_transformer_forward(n_heads, mask_type):
    transformer = MultiScaleTransformer(
        n_heads=n_heads, 
        emb_dim=256, 
        emb_num=10, 
        depth=5, 
        num_scales=4,
        block_size=8)
    sampler = HierarchicalSequentialSampler(transformer)
    xs = generate_xs(batch_size=4)
    
    seq_toks, seq_inds = sampler.random_windows(xs)

    token_counts = [8, 16, 32, 64]
    for ind, pos_emb in enumerate(transformer.pos_embs):
        assert pos_emb.weight.shape[0] == token_counts[ind]

    mask = None
    if mask_type == 'causal':
        _, mask = get_causal_mask(4 * 8)

    y = transformer(seq_toks, inds=seq_inds, mask=mask)

    assert y.shape == (4, 4, 8, 10)


def test_ms_transformer_forward():
    transformer = MultiScaleTransformer(
        n_heads=4, 
        emb_dim=256, 
        emb_num=10, 
        depth=5, 
        num_scales=4,
        block_size=8)
    sampler = HierarchicalSequentialSampler(transformer)
    xs = generate_xs(batch_size=1)
    _, mask = get_causal_mask(4 * 8)
    xs = sampler.sequential_sample_resolution(
        xs, 
        verbose=True, 
        sample=False,
        mask=mask,
        level=0,
        top_k=5)
