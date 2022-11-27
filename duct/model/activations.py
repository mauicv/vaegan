import torch


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)