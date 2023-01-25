import torch
from torch.nn import functional as F
from tqdm import tqdm


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
    def __init__(self, model, scale):
        self.model = model
        self.scale = scale
        self.data_shapes = []
        for i in range(self.model.num_scales):
            res_seq_len = self.model.block_size * (self.scale ** i)
            self.data_shapes.append(res_seq_len)

    def sample_inds(self):
        inds = torch.randint(0, self.scale, (self.model.num_scales, ))
        inds[0] = 0
        return inds, self.scaled_cumsum(inds)

    def scaled_cumsum(self, inds):
        new_vals = inds.clone()
        for i in range(1, len(inds)):
            new_vals[i] = self.scale * new_vals[i-1] + inds[i] * self.model.block_size
        return new_vals

    def sub_sample(self, xs):
        inds, seq_inds = self.sample_inds()
        tok_seqs = torch.zeros(
            xs[0].shape[0], len(xs), 
            self.model.block_size, 
            dtype=torch.long, 
            device=xs[0].device
        )
        for i, ind in enumerate(seq_inds):
            tok_seqs[:, i] = xs[i][:, ind:ind+self.model.block_size]
        return inds, seq_inds, tok_seqs

    def generate_random_xs(self, batch_size=1):
        xs = []
        if next(self.model.parameters()).is_cuda: 
            device = 'cuda'
        else:
            device = 'cpu'
        for i in range(self.model.num_scales):
            xs.append(torch.randint(
                0, self.model.emb_num, 
                (batch_size, self.data_shapes[i]), 
                dtype=torch.long,
                device=device
            ))
        return xs

    @torch.no_grad()
    def _sample(self, xs, top_k=50, temperature=0.1, sample=True):
        assert top_k <= self.model.emb_num, \
            f"top_k must be less than number of embeddings, {self.model.emb_num}"
        self.model.eval()
        inds, seq_inds, tok_seq = self.sub_sample(xs)
        logits = self.model(tok_seq, inds)
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
        for i, ind in enumerate(seq_inds):
            xs[i][:, ind:ind+self.model.block_size] = x[:, i, :]
        
        return xs

    @torch.no_grad()
    def simple_sample(
            self,
            xs,
            top_k=50, 
            iterations=10, 
            temperature=0.1, 
            sample=True, 
            verbose=False
        ):
        self.model.eval()
        for _ in tqdm(range(iterations), disable=not verbose):
            xs = self._sample(xs, top_k, temperature, sample)
        return xs
