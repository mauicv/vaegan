import torch
import numpy as np


def scale(mask_inds, factor):
    h, w = mask_inds.dims
    inds = (torch.cat((
        (mask_inds.inds % h)[:, None], 
        (mask_inds.inds // w)[:, None]
        ), dim = 1
    ) * factor).long()
    h, w = int(h * factor), int(w * factor)
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

def make_sqr_mask_ul(h, w, sqr_h, sqr_w):
    sqr = torch.zeros(h, w)
    sqr[0:sqr_h, 0:sqr_w] = 1
    return sqr.flatten()

def make_sqr_mask_br(h, w, sqr_h, sqr_w):
    sqr = torch.zeros(h, w)
    sqr[h - sqr_h:, w - sqr_w:] = 1
    inds = (torch.arange(0, h*w) - 1) % (h * w)
    return sqr.flatten()[inds]

def to_mask(mask_inds, mask_dims, mask='ul'):
    h, w = mask_inds.dims
    inds = torch.arange(0, h*w)[None, :] - mask_inds.inds[:, None]
    inds = inds % (h*w)
    if isinstance(mask, str) or mask is None:
        mask_h, mask_w = mask_dims
        sqr = {
            'ul': make_sqr_mask_ul,
            'br': make_sqr_mask_br,
            None: make_sqr_mask_ul
        }[mask](h, w, mask_h, mask_w)
        return sqr[inds]
    else:
        return mask[inds]


def to_inds(mask_inds, mask_dims, mask='ul'):
    sqr = to_mask(mask_inds, mask_dims, mask=mask)
    # print(sqr.reshape(-1, *mask_inds.dims))
    return torch.nonzero(sqr)[:, 1] \
        .reshape(mask_inds.batch, -1)


def snap(mask_inds, mins_h, mins_w, maxs_h, maxs_w):
    assert mins_h <= maxs_h
    assert mins_w <= maxs_w
    assert mins_h >= 0 and mins_w >= 0
    inds_2d = mask_inds.to_2d()
    mins = torch.ones_like(inds_2d) * torch.tensor([mins_h, mins_w])
    maxs = torch.ones_like(inds_2d) * torch.tensor([maxs_h, maxs_w])
    inds_2d = torch.max(inds_2d, mins)
    inds_2d = torch.min(inds_2d, maxs)
    return to_1d(inds_2d, *mask_inds.dims)


class MaskIndex2D:
    def __init__(self, dims, inds, batch=None):
        self._dims = np.array(dims)
        if batch is None:
            batch, *_ = inds.shape
        self.batch = batch
        self._inds = inds

    @property
    def l(self):
        return np.prod(self._dims)

    @property
    def inds(self):
        return self._inds % self.l

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

    def scale(self, factor):
        return scale(self, factor)

    def __add__(self, other):
        a = to_2d(self)
        b = to_2d(other)
        return to_1d(a + b, *self._dims)

    def __repr__(self):
        return (f'<{self.__class__.__name__} '
                f'dims={tuple(self._dims)}, '
                f'batch={self.batch}>')

    def to_mask(self, mask_dims, mask=None):
        return to_mask(self, mask_dims, mask=mask)

    def to_inds(self, mask_dims, mask=None):
        return to_inds(self, mask_dims, mask=mask)

    def snap(self, mins_h, mins_w, maxs_h, maxs_w):
        return snap(self, mins_h, mins_w, maxs_h, maxs_w)

    def next(self):
        return MaskIndex2D(
            dims=self._dims,
            inds=self._inds + 1,
            batch=self.batch
        )