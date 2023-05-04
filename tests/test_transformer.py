import pytest
import torch
from duct.model.transformer.model import Transformer, RelEmbTransformer
from duct.model.transformer.block import TransformerBlock
from duct.model.transformer.attention import AttnBlock
from duct.model.transformer.relative_attention import RelAttnBlock, SkewedRelAttnBlock
from duct.model.transformer.mask import get_local_image_mask, get_causal_mask


@pytest.mark.parametrize("n_heads", [1, 2, 4, 8])
def test_attn_block(n_heads):
    attn_block = AttnBlock(emb_dim=64, block_size=128, n_heads=n_heads)
    x = torch.randn(64, 128, 64)
    y = attn_block(x)
    assert y.shape == x.shape

@pytest.mark.parametrize("n_heads", [1, 2, 4, 8])
def test_attn_block_infer(n_heads):
    attn_block = AttnBlock(emb_dim=64, block_size=128, n_heads=n_heads)
    x = torch.randn(64, 1, 64)
    pk, pv = None, None
    for i in range(128):
        y, pk, pv = attn_block.infer(x, prev_k=pk, prev_v=pv)
        assert y.shape == x.shape
        assert pk.shape == (64, n_heads, i + 1, int(64/n_heads))
        assert pv.shape == (64, n_heads, i + 1, int(64/n_heads))


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
def test_transformer_block_infer(n_heads):
    transformer = TransformerBlock(n_heads=n_heads, block_size=128, emb_dim=64)
    x = torch.randn(64, 1, 64)
    pk, pv = None, None
    for i in range(128):
        y, pk, pv = transformer.infer(x, prev_k=pk, prev_v=pv)
        assert y.shape == x.shape
        assert pk.shape == (64, n_heads, i + 1, int(64/n_heads))
        assert pv.shape == (64, n_heads, i + 1, int(64/n_heads))


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
@pytest.mark.parametrize("trainable_pos_embeddings", [True, False])
def test_transformer_infer(n_heads, trainable_pos_embeddings):
    transformer = Transformer(
        n_heads=n_heads, 
        emb_dim=256, 
        emb_num=10, 
        depth=5, 
        block_size=128, 
        trainable_pos_embeddings=trainable_pos_embeddings)
    x = torch.randint(0, 10, (64, 1))
    pk, pv = None, None
    for i in range(128):
        y, pk, pv = transformer.infer(x, prev_ks=pk, prev_vs=pv)
        assert y.shape == (64, 1, 10)
        x = y.argmax(dim=-1)
        for j in range(5):
            assert pk[j].shape == (64, n_heads, i + 1, int(256/n_heads))
            assert pv[j].shape == (64, n_heads, i + 1, int(256/n_heads))


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
