import numpy as np
from random import seed
import torch


def set_seeds():
    np.random.seed(8)
    torch.manual_seed(8)
    torch.cuda.manual_seed(8)
    # Python std lib random seed
    seed(0)
    # Numpy, tensorflow
    np.random.seed(0)
    # Pytorch
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def disable_dropout(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
