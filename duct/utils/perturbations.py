from torch.distributions.bernoulli import Bernoulli
import torch


def perturb_seq(seq, lb, ub, p=0.5):
    seq_cpy = seq.clone()
    m = Bernoulli(torch.tensor(p, device=seq.device))
    ind = m.sample(seq_cpy.shape)
    random_tokens = torch.randint(
        lb, ub, 
        seq_cpy[ind == 0].shape, 
        device=seq.device)
    seq_cpy[ind == 0] = random_tokens
    return seq_cpy