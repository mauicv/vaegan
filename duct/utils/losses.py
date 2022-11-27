import torch.nn.functional as F
from torch.autograd import grad
import torch


def KL_div_loss(mu, logvar):
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    return torch.sum(KLD_element).mul_(-0.5)


def hinge_d_loss(real_logits, fake_logits):
    loss_real = torch.mean(F.relu(1. - real_logits))
    loss_fake = torch.mean(F.relu(1. + fake_logits))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def g_loss(fake_logits):
    return - fake_logits.mean()


def adaptive_loss(r_loss, g_loss, autoencoder):
    last_layer = autoencoder.decoder.out_conv.weight
    r_grads = grad(r_loss, last_layer, retain_graph=True)[0]
    g_grads = grad(g_loss, last_layer, retain_graph=True)[0]
    d_weight = torch.norm(r_grads) / (torch.norm(g_grads) + 1e-4)
    return 0.8 * torch.clamp(d_weight, 0.0, 1e4).detach()
