import torch
from torch.nn import functional as F
from tqdm import tqdm
from duct.model.samplers.util import top_k_logits


class HierarchicalSequentialSampler:
    def __init__(self, model, ratio=2):
        self.model = model
        self.device = next(model.parameters()).device
        self.block_size = model.block_size
        self.ratio = ratio

    @torch.no_grad()
    def sequential_sample_resolution(
            self,
            xs,
            temperature=1.0, 
            top_k=50, 
            sample=True, 
            mask=None, 
            verbose=False,
            level=0
        ):
        self.model.eval()

        for k, seq_toks, seq_inds in tqdm(
                self.windows(xs, level=level), 
                disable=not verbose):
            logits = self.model(seq_toks, inds=seq_inds, mask=mask)
            logits = logits[:, level, min(k - 1, self.block_size - 1), :] \
                / temperature

            if top_k is not None:
                logits = top_k_logits(logits, top_k)
            probs = F.softmax(logits, dim=-1)
            if sample:
                ix = torch.multinomial(probs.squeeze(), num_samples=1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            xs[level].reshape(1, -1)[:, k] = ix
        return xs

    def windows(self, xs, level=0):
        b = xs[0].shape[0]
        seq_toks = torch.zeros(
            b, self.model.num_scales, self.block_size,
            dtype=torch.long,
            device=self.device
        )
        seq_inds = torch.zeros(
            b, self.model.num_scales, self.block_size,
            dtype=torch.long,
            device=self.device
        )
        for k in range(0, xs[level].shape[-1]):
            k = torch.tensor([k], device=self.device)
            b, _ = xs[level].shape
            assert b == 1
            for ind, x in enumerate(xs):
                x_inds = torch.arange(0, self.block_size, device=self.device)[None]
                k_ind = min(max(k * 2 ** ind, self.block_size), xs[ind].shape[-1]) 
                x_inds = x_inds + k_ind - self.block_size
                seq_toks[:, ind, :] = \
                    x[:, x_inds]
                seq_inds[:, ind, :] = x_inds
            yield k, seq_toks, seq_inds

    def random_windows(self, xs):
        b, l = xs[-1].shape
        s = len(xs)
        seq_toks = torch.zeros(
            b, self.model.num_scales, self.block_size,
            dtype=torch.long,
            device=self.device
        )
        seq_inds = torch.zeros(
            b, self.model.num_scales, self.block_size,
            dtype=torch.long,
            device=self.device
        )
        inds = torch.randint(0, l, (b, 1), device=self.device)
        for ind, x in enumerate(reversed(xs), 1):
            inds = torch.max(inds, torch.tensor(self.block_size))
            k_inds = torch.arange(-self.block_size, 0, device=self.device)[None]
            k_inds = k_inds + inds
            seq_toks[:, s - ind, :] = torch.gather(x, 1, k_inds)
            seq_inds[:, s - ind, :] = k_inds
            inds = inds // self.ratio
        return seq_toks, seq_inds
