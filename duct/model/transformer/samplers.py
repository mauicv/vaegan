import torch
from torch.nn import functional as F
from tqdm import tqdm
# from duct.model.transformer.model import MultiScaleTransformer
from duct.utils.mask_inds_2d import MaskIndex2D


def top_k_logits(logits, k):
    v, _ = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[..., [-1]]] = -float('Inf')
    return out


@torch.no_grad()
def sample_sequential(
        model, 
        x, 
        temperature=1.0, 
        top_k=50, 
        sample=True, 
        mask=None, 
        verbose=False
    ):
    l = x.shape[0]
    model.eval()
    block_size = model.block_size
    seq = torch.zeros(block_size, dtype=torch.long, device=x.device)
    seq[:l] = x
    seq = seq[None, :]

    for k in tqdm(range(l, block_size), disable=not verbose):
        logits = model(seq, mask=mask)
        logits = logits[:, k-1, :] / temperature
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        probs = F.softmax(logits, dim=-1)
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        seq[0, k] = ix

    return seq[0]


@torch.no_grad()
def sample_step(
        model, 
        x, 
        top_k=50, 
        iterations=10, 
        temperature=0.1, 
        sample=True, 
        verbose=False
    ):
    model.eval()
    for _ in tqdm(range(iterations), disable=not verbose):
        logits = model(x)
        logits = logits / temperature
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        probs = F.softmax(logits, dim=-1)
        b_dim, s_dim, _ = logits.shape
        if sample:
            probs = probs.cumsum(-1)
            rns = torch.rand(b_dim, s_dim, 1, device=x.device)
            x = torch.searchsorted(probs, rns)
            x = x.squeeze(-1)
        else:
            _, x = torch.topk(probs, k=1, dim=-1)
            x = x.squeeze(-1)
    return x


class HierarchySampler:
    def __init__(self, model):
        self.model = model

    def sub_sample(self, xs):
        batch = xs[0].shape[0]
        device = xs[0].device
        shapes = [x.shape[1:] for x in xs]
        layer_sample_shape = shapes[0]
        masks = [MaskIndex2D.random(batch, dims=layer_sample_shape)]
        for _ in range(3):
            masks.append(masks[-1].upscale(2).perturb(layer_sample_shape))
        masks = [m.to_inds(layer_sample_shape) for m in masks]
        xs_sub = [x.flatten()[m.flatten()].reshape(batch, -1)
                  for x, m in zip(xs, masks)]
        xs_sub = torch.cat([x[:, None, :] for x in xs_sub], dim=1)
        mask_inds = torch.cat([m[:, None, :] for m in masks], dim=1)
        mask_inds = mask_inds.to(device)
        xs_sub = xs_sub.to(device)
        return mask_inds, xs_sub

    @torch.no_grad()
    def _sample(
            self,
            xs,
            top_k=50,
            temperature=0.1,
            sample=True,
            layers=None
        ):
        assert xs[0].shape[0] == 1, "batch size must be 1"
        assert top_k <= self.model.emb_num, \
            f"top_k must be less than number of embeddings, {self.model.emb_num}"
        self.model.eval()
        seq_inds, tok_seq = self.sub_sample(xs)
        logits = self.model(tok_seq, seq_inds)
        logits = logits / temperature

        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        probs = F.softmax(logits, dim=-1)
        b, s, l, _ = logits.shape

        if sample:
            probs = probs.cumsum(-1)
            rns = torch.rand(b, s, l, 1, device=xs[0].device)
            x = torch.searchsorted(probs, rns)
            x = x.squeeze(-1)
        else:
            _, x = torch.topk(probs, k=1, dim=-1)
            x = x.squeeze(-1)
        
        seqs = []
        for layer, (x_res, ind, toks) in enumerate(zip(xs, seq_inds[0], x[0])):
            _, h, w = x_res.shape
            if layer in layers:
                x_res = x_res[0]
                x_res = x_res.reshape(-1)
                x_res = x_res.scatter(0, ind, toks)
            seqs.append(x_res.reshape(1, h, w))

        return seqs

    @torch.no_grad()
    def simple_sample(
            self,
            xs,
            top_k=50, 
            iterations=10, 
            temperature=0.1, 
            sample=True, 
            verbose=False,
            layers=None
        ):
        self.model.eval()
        for _ in tqdm(range(iterations), disable=not verbose):
            xs = self._sample(xs, top_k, temperature, sample, layers=layers)
        return xs
