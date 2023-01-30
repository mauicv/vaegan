import torch


def scaled_cumsum(inds, scale=4, width=64):
    new_vals = inds.clone()
    for i in range(1, len(inds)):
        new_vals[i] = scale * new_vals[i-1] + inds[i] * width
    return new_vals


def test_to_seq():
    print()
    tok_seqs = torch.zeros(24, 4, 64)
    with torch.no_grad():
        inds = torch.randint(0, 4, (4, ))
        inds[0] = 0
        scaled_inds = scaled_cumsum(inds)
        for i, shape in enumerate([
                (24, 8, 8), 
                (24, 16, 16), 
                (24, 32, 32), 
                (24, 64, 64)
            ]):
            tok_seq = torch.randint(0, 25, shape)
            b, h, w = tok_seq.shape
            tok_seq = tok_seq.reshape(b, h * w)
            tok_seq = tok_seq[:, scaled_inds[i]:scaled_inds[i] + 64]
            tok_seqs[:, i] = tok_seq

    print(tok_seqs.shape)
