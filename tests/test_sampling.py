import torch
import pytest
from duct.model.transformer.model import Transformer
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
