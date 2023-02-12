import matplotlib.pyplot as plt
import pytest
from duct.utils.mask_inds_2d import MaskIndex2D
import torch


def test_transforms():
    mask_inds_1 = MaskIndex2D.random(4, (8, 8))
    ind_2d = mask_inds_1.to_2d()
    mask_inds_2 = MaskIndex2D.from_2d(ind_2d, (8, 8))
    assert torch.all(mask_inds_2.inds==mask_inds_1.inds)


def test_addition():
    a = MaskIndex2D.random(4, (8, 8))
    b = MaskIndex2D.random(4, (8, 8))
    c_1 = a + b
    a_2d = a.to_2d()
    b_2d = b.to_2d()
    c_2 = MaskIndex2D.from_2d(a_2d + b_2d, (8, 8))
    assert torch.all(c_1.inds==c_2.inds)


def test_perturbation():
    # cook ind values so that perturbations can't go out of bounds
    # This means we don't have to worry about torality.
    t = torch.tensor([[2, 2], [3, 3]])
    a = MaskIndex2D.from_2d(t, (8, 8))
    a_p = a.perturb((2, 2))
    assert torch.all(a.inds < 8*8)
    a = a.to_2d()
    a_p = a_p.to_2d()
    assert torch.all(a_p - a < 2)


def test_scale():
    a = MaskIndex2D.random(4, (8, 8))
    b = a.scale(2)
    a = a.to_2d()
    b = b.to_2d()
    assert torch.all(2*a == b)

def test_snap():
    a = MaskIndex2D.random(4, (8, 8))
    d = a.snap(3, 2, 5, 6)
    inds_2d = d.to_2d()
    assert torch.all(inds_2d[:, 0] >= 3)
    assert torch.all(inds_2d[:, 1] >= 2)
    assert torch.all(inds_2d[:, 0] <= 5)
    assert torch.all(inds_2d[:, 1] <= 6)


@pytest.mark.skip(reason="Displays plots to screen")
def test_plot_masks():
    masks = [MaskIndex2D.random(4, dims=(8, 8))]
    for i in range(3):
        masks.append(masks[-1].scale(2).perturb((4, 4)))
    masks = [m.to_mask((4, 4)).reshape(-1, *m.dims) for m in masks]

    _, axs = plt.subplots(ncols=4, nrows=4)
    for b in range(4):
        for i, mask in enumerate(masks):
            axs[b, i].imshow(mask[b])
    plt.show()