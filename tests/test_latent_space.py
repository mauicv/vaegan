import torch
from duct.model.latent_spaces import LinearLatentSpace, StochasticLinearLatentSpace, \
    StochasticLatentSpace, VQLatentSpace1D, VQLatentSpace2D


def test_linear_latent_space():
    latent_space = LinearLatentSpace((64, 8, 8), (64, 8, 8), 5)
    in_tensor = torch.randn((64, 64, 8, 8))
    out_tensor, *_ = latent_space(in_tensor)
    assert in_tensor.shape == out_tensor.shape


def test_stochastic_linear_latent_space():
    latent_space = StochasticLinearLatentSpace((64, 8, 8), (64, 8, 8), 5)
    in_tensor = torch.randn((64, 64, 8, 8))
    out_tensor, mu, logvar = latent_space(in_tensor)
    assert in_tensor.shape == out_tensor.shape
    assert mu.shape == (64, 5)
    assert logvar.shape == (64, 5)


def test_stochastic_latent_space():
    latent_space = StochasticLatentSpace((64, 8, 8), (64, 8, 8))
    in_tensor = torch.randn((64, 64, 8, 8))
    out_tensor, mu, logvar = latent_space(in_tensor)
    assert in_tensor.shape == out_tensor.shape
    assert mu.shape == out_tensor.shape
    assert logvar.shape == out_tensor.shape


def test_vq_latent_space_2d():
    latent_space = VQLatentSpace2D(num_embeddings=10, embedding_dim=64, commitment_cost=1)
    in_tensor = torch.randn((64, 64, 2, 2))
    output, loss, _, encoded = latent_space(in_tensor)
    assert output.shape == (64, 64, 2, 2)
    assert loss.shape == ()
    assert encoded.shape == (64, 2, 2, 10)


def test_vq_latent_space_usage_reset_2d():
    torch.manual_seed(0)
    num_embeddings = 256
    latent_space = VQLatentSpace2D(num_embeddings=num_embeddings, embedding_dim=64, commitment_cost=1)
    in_tensor = torch.randn((16, 64, 8, 8))
    _, _, _, _ = latent_space(in_tensor)
    missed_indcies = [k for k, v in latent_space.usage_counts.items() if v < 100]
    assert len(missed_indcies) > 0
    vectors_1 = latent_space._embedding.weight[missed_indcies]
    for _ in range(99):
        _, _, _, _ = latent_space(in_tensor)

    for ind in missed_indcies:
        assert latent_space.usage_counts[ind] == 100

    vectors_2 = latent_space._embedding.weight[missed_indcies]
    assert not torch.all(vectors_1 == vectors_2)


def test_vq_latent_space_1d():
    latent_space = VQLatentSpace1D(num_embeddings=10, embedding_dim=64, commitment_cost=1)
    in_tensor = torch.randn((64, 64, 8))
    output, loss, _, encoded = latent_space(in_tensor)
    assert output.shape == (64, 64, 8)
    assert loss.shape == ()
    assert encoded.shape == (64, 8, 10)
