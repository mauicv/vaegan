import math
import torch


def make_tile_index(d):
    s = torch.arange(0, d*d)
    p = torch.tensor([
        [1, 0, 0, 0], 
        [0, 0, 1, 0], 
        [0, 1, 0, 0], 
        [0, 0, 0, 1]
    ])
    a = int(math.log(d, 2)) - 1
    for i in reversed([_ + 1 for _ in range(a)]):
        for j in range(i):
            s = p @ s.reshape(-1, 4, 2**(i + j))

    return s.reshape(-1)