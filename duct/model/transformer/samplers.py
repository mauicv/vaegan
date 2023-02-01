import torch
from torch.nn import functional as F
from tqdm import tqdm
from duct.model.transformer.model import MultiScaleTransformer


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

    def make_grid_inds(self, w, h):
        add = ((i, j) for i in range(0, int(w)) for j in range(0, int(h)))
        return torch.tensor(list(add)).reshape(w, h, 2)

    def sub_sample(self, xs, batch_size=1):
        b, w, h = xs[0].shape
        add = self.make_grid_inds(w, h).to(xs[0].device)
        inds = []
        for i in range(0, self.model.num_scales):
            if i == 0:
                ind = torch.zeros((batch_size, 2), dtype=torch.long, device=xs[0].device)
            else:
                rnd = torch.randint(0, int(w), (batch_size, 2), device=xs[0].device)
                ind = last_inds * 2 + rnd
            last_inds = ind
            ind = ind[:, None, None] + add[None, :]
            offset_mult = torch.tensor([1, w*(2**i)]).to(xs[0].device)
            ind = (ind.reshape(b, -1, 2) * offset_mult).sum(-1)
            inds.append(ind[:, None, :])

        xs = [x.reshape(b, -1) for x in xs]
        seq_inds = torch.cat(inds, dim=1).to(xs[0].device)
        tok_seqs = torch.zeros(
            b, len(xs),
            self.model.block_size, 
            dtype=torch.long, 
            device=xs[0].device
        )
        for i, ind in enumerate(seq_inds.permute(1, 0, 2)):
            tok_seqs[:, i] = xs[i].gather(1, ind)
        return seq_inds, tok_seqs

    @torch.no_grad()
    def _sample(self, xs, top_k=50, temperature=0.1, sample=True, layers=None):
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
        for layer, (x_res, ind, toks) in enumerate(zip(xs, seq_inds[0], tok_seq[0])):
            x_res = x_res[0]
            h, w = x_res.shape
            x_res = x_res.reshape(-1)
            if layer in layers:
                x_res.scatter_(0, ind, toks)
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
        if layers is None:
            layers = set(range(self.model.num_scales))
        elif isinstance(layers, list):
            layers = set(layers)

        for _ in tqdm(range(iterations), disable=not verbose):
            xs = self._sample(xs, top_k, temperature, sample, layers=layers)
        return xs
