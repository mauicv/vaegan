import torch


def swish(x):
    return x*torch.sigmoid(x)


def get_nonlinearity(type='ELU'):
    return {
        'swish': lambda: swish,
        'relu': lambda: torch.nn.ReLU(),
        'leakyrelu': lambda: torch.nn.LeakyReLU(0.2),
        'tanh': lambda: torch.nn.Tanh(),
        'sigmoid': lambda: torch.nn.Sigmoid(),
        'ELU': lambda: torch.nn.ELU(),
    }[type]()
    
    