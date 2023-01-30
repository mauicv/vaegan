import torch
from duct.model.transformer.model import Transformer, MultiScaleTransformer
from duct.model.transformer.samplers import sample_sequential, sample_step, HierarchySampler
from duct.model.transformer.mask import get_local_image_mask


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

def test_hierarchy_sampler_sample_inds():
    transformer = MultiScaleTransformer(
        n_heads=4, 
        emb_dim=256, 
        emb_num=10, 
        depth=5, 
        num_scales=4,
        block_size=128
    )
    sampler = HierarchySampler(transformer)
    assert sampler.data_shapes == [128, 512, 2048, 8192]
    inds = sampler.sample_inds()
    assert inds.shape == (4, )
    for i, ub in zip(inds, sampler.data_shapes):
        assert 0 <= i <= ub - 128


def test_hierarchy_sampler_sub_sample():
    transformer = MultiScaleTransformer(
        n_heads=4, 
        emb_dim=256, 
        emb_num=10, 
        depth=5, 
        num_scales=4,
        block_size=128
    )
    sampler = HierarchySampler(transformer)
    assert sampler.data_shapes == [128, 512, 2048, 8192]
    x = [
        torch.ones((24, 128)),
        torch.ones((24, 512)),
        torch.ones((24, 2048)),
        torch.ones((24, 8192)),
    ]
    inds, encs = sampler.sub_sample(x)
    assert inds.shape == (4, )
    assert encs.shape == (24, 4, 128)
    assert torch.all(encs == 1)


def test_hierarchy_sampler_random_xs():
    transformer = MultiScaleTransformer(
        n_heads=4, 
        emb_dim=256, 
        emb_num=10, 
        depth=5, 
        num_scales=4,
        block_size=128
    )
    sampler = HierarchySampler(transformer)
    xs = sampler.generate_random_xs()
    for x, s in zip(xs, sampler.data_shapes):
        assert x.shape == (1, s)


def test_hierarchy_sampler__sample():
    transformer = MultiScaleTransformer(
        n_heads=4, 
        emb_dim=256, 
        emb_num=10, 
        depth=5, 
        num_scales=4,
        block_size=128
    )
    sampler = HierarchySampler(transformer)
    xs = sampler.generate_random_xs()
    shapes_1 = [x.shape for x in xs]
    xs = sampler.simple_sample(xs, top_k=5, iterations=2, sample=True)
    shapes_2 = [x.shape for x in xs]
    assert shapes_1 == shapes_2
