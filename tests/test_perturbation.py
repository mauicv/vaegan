from duct.utils.perturbations import perturb_seq
import torch
import pytest
import numpy as np


@pytest.mark.parametrize("p", [0.1, 0.2, 0.3, 0.4, 0.5])
def test_perturb_seq(p):
    seq1 = torch.zeros((16, 1024), dtype=torch.long)
    seq2 = perturb_seq(seq1, 0, 256, p=p)
    assert not torch.all(seq1 == seq2)
    assert np.isclose((seq2 == 0).sum() / (16 * 1024), p, atol=0.1)

@pytest.mark.parametrize("p", [0.1, 0.2, 0.3, 0.4, 0.5])
def test_perturb_seq_ms(p):
    seq1 = torch.zeros((16, 4, 128), dtype=torch.long)
    seq2 = perturb_seq(seq1, 0, 256, p=p)
    assert not torch.all(seq1 == seq2)
    assert np.isclose((seq2 == 0).sum() / (16 * 128 * 4), p, atol=0.1)
