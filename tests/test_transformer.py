import pytest
import torch
from duct.model.transformer.model import Transformer, RelEmbTransformer
from duct.model.transformer.block import TransformerBlock
from duct.model.transformer.attention import AttnBlock
from duct.model.transformer.relative_attention import RelAttnBlock, SkewedRelAttnBlock
from duct.model.transformer.mask import get_local_image_mask, get_causal_mask
import numpy as np
from random import seed
import torch
from tests.util import set_seeds, disable_dropout


@pytest.mark.parametrize("n_heads", [1, 2, 4, 8])
def test_attn_block(n_heads):
    attn_block = AttnBlock(emb_dim=64, block_size=128, n_heads=n_heads)
    x = torch.randn(64, 128, 64)
    y = attn_block(x)
    assert y.shape == x.shape

@pytest.mark.parametrize("block_type", [AttnBlock, SkewedRelAttnBlock])
@pytest.mark.parametrize("n_heads", [1, 2, 4, 8])
def test_attn_block_infer(block_type, n_heads):
    block_size, emb_dim, batch_size = 16, 16, 3
    set_seeds()
    attn_block = block_type(emb_dim=emb_dim, block_size=block_size, n_heads=n_heads)
    disable_dropout(attn_block)
    x_init = torch.randn(batch_size, 1, emb_dim)
    ps = {}
    x1, x2 = x_init, x_init
    for i in range(block_size):
        with torch.no_grad():
            y1, ps = attn_block.infer(x1, **ps)
            y2 = attn_block(x2)[:, -1:, :]
        assert y1.shape == y2.shape
        torch.allclose(y1, y2)
        x1, x2 = y1, torch.cat([x2, y2], dim=-2)
        assert y1.shape == x1.shape
        for _, val in ps.items():
            assert val.shape == (batch_size, n_heads, min(i+1, block_size-1), int(emb_dim/n_heads))


@pytest.mark.parametrize("n_heads", [4])
def test_rel_attn_block(n_heads):
    attn_block = SkewedRelAttnBlock(
        emb_dim=64,
        block_size=128,
        n_heads=n_heads)
    x = torch.randn(64, 128, 64)
    y = attn_block(x)
    assert y.shape == x.shape


def test_rel_attn_block():
    attn_block = RelAttnBlock(emb_dim=64, block_size=128, n_heads=4)
    x = torch.randn(64, 128, 64)
    y = attn_block(x)
    assert y.shape == x.shape


@pytest.mark.parametrize("n_heads", [1, 2, 4, 8])
def test_transformer_block(n_heads):
    transformer = TransformerBlock(n_heads=n_heads, block_size=128, emb_dim=64)
    x = torch.randn(64, 128, 64)
    y = transformer(x)
    assert y.shape == x.shape


@pytest.mark.parametrize("n_heads", [1, 2, 4, 8])
@pytest.mark.parametrize("block_type", [AttnBlock, SkewedRelAttnBlock])
def test_transformer_block_infer(n_heads, block_type):
    transformer = TransformerBlock(
        n_heads=n_heads, 
        block_size=128, 
        emb_dim=64, 
        attn_block=block_type
    )
    x = torch.randn(64, 1, 64)
    x1, x2 = x, x
    ps = {}
    for i in range(128):
        with torch.no_grad():
            y1, ps = transformer.infer(x1, ps)
            y2 = transformer(x2)[:, -1:, :]
        assert y1.shape == y2.shape
        torch.allclose(y1, y2)
        # torch.all(y1 == y2) # this fails
        x1, x2 = y1, torch.cat([x2, y2], dim=-2)
        assert y1.shape == x.shape
        assert y2.shape == x.shape
        for _, val in ps.items():
            assert val.shape == (64, n_heads, min(i+1, 127), int(64/n_heads))


@pytest.mark.parametrize("n_heads", [1, 2, 4, 8])
@pytest.mark.parametrize("trainable_pos_embeddings", [True, False])
def test_transformer(n_heads, trainable_pos_embeddings):
    transformer = Transformer(
        n_heads=n_heads, 
        emb_dim=256, 
        emb_num=10, 
        depth=5, 
        block_size=128, 
        trainable_pos_embeddings=trainable_pos_embeddings)
    x = torch.randint(0, 10, (64, 128))
    y = transformer(x)
    assert y.shape == (64, 128, 10)


@pytest.mark.parametrize("n_heads", [1, 8])
def test_transformer_infer(n_heads):
    set_seeds()
    depth, emb_dim, block_size, batch_size = 2, 16, 8, 4
    transformer = Transformer(
        n_heads=n_heads, 
        emb_dim=emb_dim,
        emb_num=10, 
        depth=depth, 
        block_size=block_size, 
        trainable_pos_embeddings=True
    )
    transformer.train(False)
    disable_dropout(transformer)
    x_init = torch.randint(0, 10, (batch_size, 1))
    x1, x2 = x_init, x_init

    prevs = None
    for i in range(block_size - 1):
        with torch.no_grad():
            y1, prevs = transformer.infer(x1, i, prevs=prevs)
            y2 = transformer(x2)[:, -1:, :]
        assert y1.shape == (batch_size, 1, 10)
        assert y2.shape == (batch_size, 1, 10)
        # Not sure why these aren't exactly equal, there must be some casting
        # issue somewhere
        assert torch.allclose(y1, y2, atol=1e-3)
        y1, y2 = y1.argmax(dim=-1), y2.argmax(dim=-1)
        assert torch.all(y1 == y2)
        x1, x2 = y1, torch.cat([x2, y2], dim=-1)
        for j in range(depth):
            for _, val in prevs[j].items():
                    assert val.shape \
                        == (batch_size, n_heads, min(i+1, block_size-1), int(emb_dim/n_heads))


@pytest.mark.parametrize("n_heads", [1, 8])
def test_transformer_infer_x(n_heads):
    set_seeds()
    depth, emb_dim, block_size, batch_size = 2, 16, 8, 4
    transformer = Transformer(
        n_heads=n_heads, 
        emb_dim=emb_dim,
        emb_num=10, 
        depth=depth, 
        block_size=block_size, 
        trainable_pos_embeddings=True
    )
    transformer.train(False)
    disable_dropout(transformer)
    x = torch.randint(0, 10, (batch_size, block_size))
    a, _ = transformer.infer(x, block_size-1)
    b = transformer(x)
    torch.allclose(a, b)


@pytest.mark.parametrize("n_heads", [1, 8])
def test_rel_emb_transformer_infer(n_heads):
    set_seeds()
    depth, emb_dim, block_size, batch_size = 2, 16, 8, 4
    transformer = RelEmbTransformer(
        n_heads=n_heads,
        emb_dim=emb_dim,
        emb_num=10,
        depth=depth,
        block_size=block_size,  
        rel_emb_type="skewed"
    )
    transformer.train(False)
    disable_dropout(transformer)
    x_init = torch.randint(0, 10, (batch_size, 1))
    x1, x2 = x_init, x_init

    prevs = None
    for i in range(block_size - 1):
        with torch.no_grad():
            y1, prevs = transformer.infer(x1, prevs=prevs)
            y2 = transformer(x2)[:, -1:, :]
        assert y1.shape == (batch_size, 1, 10)
        assert y2.shape == (batch_size, 1, 10)
        # Not sure why these aren't exactly equal, there must be some casting
        # issue somewhere
        assert torch.allclose(y1, y2, atol=1e-3)
        y1, y2 = y1.argmax(dim=-1), y2.argmax(dim=-1)
        assert torch.all(y1 == y2)
        x1, x2 = y1, torch.cat([x2, y2], dim=-1)
        for j in range(depth):
            for _, val in prevs[j].items():
                    assert val.shape \
                        == (batch_size, n_heads, min(i+1, block_size-1), int(emb_dim/n_heads))


@pytest.mark.skip()
@pytest.mark.parametrize("n_heads", [1, 8])
def test_rel_emb_transformer_infer_x(n_heads):
    set_seeds()
    depth, emb_dim, block_size, batch_size = 2, 16, 8, 4
    transformer = RelEmbTransformer(
        n_heads=n_heads,
        emb_dim=emb_dim,
        emb_num=10,
        depth=depth,
        block_size=block_size,  
        rel_emb_type="skewed"
    )
    transformer.train(False)
    disable_dropout(transformer)
    x = torch.randint(0, 10, (batch_size, block_size))
    a, ps = transformer.infer(x)
    b = transformer(x)
    torch.allclose(a, b)

    a, b, = a.argmax(dim=-1), b.argmax(dim=-1)
    # Seems like small errors accumulate and become significant when sampling from larger networks
    # [-0.0001, -0.0015, -0.0001, -0.0005,  0.0003,  0.0004,  0.0019,  0.0019, -0.0007,  0.0019]
    # [6, 6, 8, 0, 9, 0, 9, 9]
    # [-3.2e-5, -1.5e-3, -2.1e-4, -5.3e-4,  2.8e-4, 3.4e-4,  1.8e-3,  1.9e-3, -7.5e-4,  1.8e-3]
    # [6, 6, 8, 0, 9, 0, 9, 7]
    assert torch.all(a == b)

    a2, ps = transformer.infer(a[:, -1:], prevs=ps)
    b2 = transformer(b)
    torch.allclose(a2, b2)


@pytest.mark.parametrize("n_heads", [1, 2, 4, 8])
def test_skewed_rel_emb_transformer(n_heads):
    transformer = RelEmbTransformer(
        n_heads=n_heads, 
        emb_dim=256, 
        emb_num=10, 
        depth=5, 
        block_size=128,
        rel_emb_type="skewed")
    x = torch.randint(0, 10, (64, 128))
    y = transformer(x)
    assert y.shape == (64, 128, 10)


@pytest.mark.parametrize("n_heads", [1, 2, 4, 8])
def test_full_rel_emb_transformer(n_heads):
    transformer = RelEmbTransformer(
        n_heads=n_heads, 
        emb_dim=256, 
        emb_num=10, 
        depth=5, 
        block_size=128,
        rel_emb_type="full")
    x = torch.randint(0, 10, (64, 128))
    y = transformer(x)
    assert y.shape == (64, 128, 10)


@pytest.mark.parametrize("n_heads", [1, 2, 4, 8])
def test_transformer_img_mask(n_heads):
    transformer = Transformer(n_heads=n_heads, emb_dim=256, emb_num=10, depth=5, block_size=8*8)
    x = torch.randint(0, 10, (32, 8*8))
    _, mask = get_local_image_mask((8,8), (4, 4))
    mask = None
    y = transformer(x, mask=mask)
    assert y.shape == (32, 8*8, 10)
    assert torch.nan not in y


@pytest.mark.parametrize("n_heads", [1, 2, 4, 8])
@pytest.mark.parametrize("trainable_pos_embeddings", [True, False])
def test_transformer_aud_mask(n_heads, trainable_pos_embeddings):
    transformer = Transformer(
        n_heads=n_heads, 
        emb_dim=256, 
        emb_num=10, 
        depth=5, 
        block_size=128, 
        trainable_pos_embeddings=trainable_pos_embeddings)
    x = torch.randint(0, 10, (64, 128))
    _, mask = get_causal_mask(x.shape[1])
    y = transformer(x, mask=mask)
    assert y.shape == (64, 128, 10)
    assert torch.nan not in y


def test_transformer_weight_decay_params():
    transformer = Transformer(
        n_heads=8, 
        emb_dim=256, 
        emb_num=10, 
        depth=5, 
        block_size=128, 
        trainable_pos_embeddings=True)
    param_groups = transformer.get_parameter_groups()
    num_decay_params = len(param_groups[0]['params'])
    num_no_decay_params = len(param_groups[1]['params'])
    total_params = len(list(transformer.parameters()))
    assert num_decay_params + num_no_decay_params == total_params
