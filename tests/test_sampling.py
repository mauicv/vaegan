import torch
import pytest
from duct.model.transformer.model import Transformer, MultiScaleTransformer
from duct.model.samplers import sample_sequential, sample_step, HierarchicalSequentialSampler
from duct.model.transformer.mask import get_local_image_mask


def generate_xs(s=8, emb_num=10, batch_size=10, device='cpu'):
    xs = []
    data_shapes = [s, s*2, s*(2**2), s*(2**3)]
    for data_shape in data_shapes:
        xs.append(torch.randint(
            0, emb_num, 
            (batch_size, data_shape),
            dtype=torch.long,
            device=device
        ))
    return xs


def test_sample_sequential():
    transformer = Transformer(n_heads=1, emb_dim=256, emb_num=10, depth=5, block_size=8*8)
    x = torch.randint(0, 10, (8, ))
    _, mask = get_local_image_mask((8, 8), (4, 4))
    y = sample_sequential(transformer, x, sample=False, mask=mask, top_k=5)
    assert torch.all(y[:x.shape[0]] == x)
    assert torch.any(y[x.shape[0]:] > 0)
    y = sample_sequential(transformer, x, sample=True, mask=mask, top_k=None)
    assert torch.all(y[:x.shape[0]] == x)
    assert torch.any(y[x.shape[0]:] > 0)


def test_sample_step():
    transformer = Transformer(n_heads=1, emb_dim=256, emb_num=10, depth=5, block_size=8*8)
    x = torch.randint(0, 10, (1, 10, ))
    sample_step(
        transformer, x, 
        top_k=5, 
        iterations=3, 
        temperature=0.1, 
        sample=True, 
        verbose=False
    )


@pytest.mark.parametrize('level', [0, 1, 2, 3])
def test_sequential_hierarchy_sampler_window(level):
    transformer = MultiScaleTransformer(
        n_heads=4, 
        emb_dim=256, 
        emb_num=10, 
        depth=5, 
        num_scales=4,
        block_size=4
    )
    sampler = HierarchicalSequentialSampler(transformer)
    xs = generate_xs(s=4, batch_size=1)
    xs = [x.reshape(1, -1) for x in xs]
    for k, seq_toks, seq_inds in sampler.windows(xs, level=level):
        assert seq_toks.shape == (1, 4, 4)
        assert seq_inds.shape == (1, 4, 4)

def test_sequential_hierarchy_sampler_window():
    transformer = MultiScaleTransformer(
        n_heads=4, 
        emb_dim=256, 
        emb_num=10, 
        depth=5, 
        num_scales=4,
        block_size=4
    )
    sampler = HierarchicalSequentialSampler(transformer)
    batch_size=2
    xs = generate_xs(s=8, batch_size=batch_size)
    seq_toks, seq_inds = sampler.random_windows(xs)
