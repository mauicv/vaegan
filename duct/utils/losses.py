import torch.nn.functional as F
from torch.autograd import grad
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram


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
    last_layer = autoencoder.decoder.output_conv.weight
    r_grads = grad(r_loss, last_layer, retain_graph=True)[0]
    g_grads = grad(g_loss, last_layer, retain_graph=True)[0]
    d_weight = torch.norm(r_grads) / (torch.norm(g_grads) + 1e-4)
    return torch.clamp(d_weight, 0.0, 1e4).detach()



class SpectralLoss(nn.Module):
    def __init__(self, window_length, sample_rate=24000, device="cuda"):
        super(SpectralLoss, self).__init__()
        self.window_length = window_length
        self.hop_length = self.window_length//4
        device = torch.device(device)
        
        self.melspec = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=self.window_length,
            hop_length=self.hop_length,
            normalized=True,
            n_mels=64,
            wkwargs={"device": device}
        ).to(device)

    def forward(self, input, target, reduction="mean"):
        S_x = self.melspec(input)
        S_G_x = self.melspec(target)
        a = (S_x - S_G_x).abs().sum(dim=(1, 2, 3))
        b = ((S_x - S_G_x)**2).sum(dim=(1, 2, 3))**0.5
        loss = a + b
        if reduction == "mean":
            loss = loss.mean()
        return loss


class MultiSpectralLoss(nn.Module):
    def __init__(
                self, 
                window_lengths=tuple(2**i for i in range(6, 12)),
                device="cuda",
                reduction="mean"
            ):
        super(MultiSpectralLoss, self).__init__()
        self.window_lengths=window_lengths
        self.spectral_losses = []
        self.reduction = reduction
        for window_length in self.window_lengths:
            spectral_loss = SpectralLoss(
                window_length=window_length,
                device=device
            )
            self.spectral_losses.append(spectral_loss)

    def forward(self, x, y):
        losses = []
        for spectral_loss in self.spectral_losses:
            loss = spectral_loss(x, y, reduction=self.reduction)
            losses.append(loss)
        if self.reduction == "mean":
            return sum(losses) / len(losses)
        return losses