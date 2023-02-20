import torch


def top_k_logits(logits, k):
    v, _ = torch.topk(logits, k, -1)
    out = logits.clone()
    out[out < v[..., [-1]]] = -float('Inf')
    return out