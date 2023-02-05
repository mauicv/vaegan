import torch
import numpy as np


def upscale(mask_inds, factor):
    h, w = mask_inds.dims
    b = mask_inds.batch

    inds = torch.cat((
        (mask_inds.inds % h)[:, None], 
        (mask_inds.inds // w)[:, None]
        ), dim = 1
    ) * factor
    h, w = h * factor, w * factor
    return to_1d(inds, h, w)


def perturb(mask_inds, bounds):
    h_b, w_b = bounds
    b = mask_inds.batch
    pert = MaskIndex2D.random(batch=b, dims=(h_b, w_b))
    return mask_inds + pert


def to_2d(mask_inds):
    h, w = mask_inds.dims
    return torch.cat((
        (mask_inds.inds % h)[:, None],
        (mask_inds.inds // w)[:, None]
        ), dim = 1)


def to_1d(inds, h, w):
    return MaskIndex2D.from_2d(dims=(h, w), inds=inds)


def to_mask(mask_inds, mask_dims):
    h, w = mask_inds.dims
    inds = torch.arange(0, h*w)[None, :] - mask_inds.inds[:, None]
    inds = inds % (h*w)
    mask_h, mask_w = mask_dims
    sqr = torch.zeros(h, w)
    sqr[0:mask_h, 0:mask_w] = 1
    sqr = sqr.flatten()
    return sqr[inds]


def to_inds(mask_inds, mask_dims):
    a, b = mask_dims
    sqr = to_mask(mask_inds, mask_dims)
    return torch.nonzero(sqr)[:, 1] \
        .reshape(-1, a * b)


class MaskIndex2D:
    def __init__(self, batch, dims, inds):
        self._dims = np.array(dims)
        self.batch = batch
        self.inds = inds % self.l

    @property
    def l(self):
        return np.prod(self._dims)

    @property
    def dims(self):
        return self._dims

    @classmethod
    def random(cls, batch, dims):
        l = np.prod(np.array(dims))
        inds = torch.randint(0, l, (batch, ))
        return cls(batch=batch, dims=dims, inds=inds)

    @classmethod
    def from_2d(cls, inds, dims):
        h, w = dims
        b, *_ = inds.shape
        inds = (inds[:, 1] * h + inds[:, 0]) % (h * w)
        return cls(batch=b, dims=(h, w), inds=inds)

    def to_2d(self):
        return to_2d(self)

    def perturb(self, bounds):
        return perturb(self, bounds)

    def upscale(self, factor):
        return upscale(self, factor)

    def __add__(self, other):
        a = to_2d(self)
        b = to_2d(other)
        return to_1d(a + b, *self._dims)

    def __repr__(self):
        return (f'<{self.__class__.__name__} '
                f'dims={tuple(self._dims)}, '
                f'batch={self.batch}>')

    def to_mask(self, mask_dims):
        return to_mask(self, mask_dims)

    def to_inds(self, mask_dims):
        return to_inds(self, mask_dims)
