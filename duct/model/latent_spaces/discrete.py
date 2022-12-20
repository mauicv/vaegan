import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch
from einops import rearrange


# Discrete Latent Spaces
class VQLatentSpace2D(nn.Module):
    """VQ_Latent_space Implementation
    Taken from https://github.com/zalandoresearch/pytorch-vq-vae
    Adapted to include a solution to codebook collapse. See: 
    https://www.reddit.com/r/MachineLearning/comments/nxjqvb/d_preventing_index_collapse_in_vqvae/h1fu9cs/
    Also from openai paper: https://arxiv.org/pdf/2005.00341.pdf
    >   VQ-VAEs are known to suffer from codebook collapse, wherein all encodings get mapped to a 
        single or few embedding vectors while the other embedding vectors in the codebook are not
        used, reducing the information capacity of the bottleneck. To prevent this, we use random
        restarts: when the mean usage of a codebook vector falls below a threshold, we randomly 
        reset it to one of the encoder outputs from the current batch
    """ # noqa
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VQLatentSpace2D, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(
            self._num_embeddings, 
            self._embedding_dim
        )
        self._embedding.weight.data \
            .uniform_(
                -1/self._num_embeddings, 
                1/self._num_embeddings)
        self._commitment_cost = commitment_cost

        self.usage_counts = {i: 100 for i in \
            range(self._num_embeddings)}

    def forward(self, inputs, training=True):
        # convert inputs from BCHW -> BHWC
        inputs = rearrange(inputs, 'b c h w -> b h w c') \
            .contiguous()
        b, h, w, c = inputs.shape
        input_shape = (b, h, w, c)

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True) 
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(
                flat_input, 
                self._embedding.weight.t()
            )
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1) \
            .unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], 
            self._num_embeddings, 
            device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        if training:
            for ind in range(self._num_embeddings):
                if ind not in encoding_indices.t().tolist()[0]:
                    self.usage_counts[ind] -= 1
                else:
                    self.usage_counts[ind] += 1

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight) \
            .view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        if training:
            for vector_ind, count in self.usage_counts.items():
                if count == 0:
                    rand_batch = np.random.randint(0, flat_input.shape[0])
                    with torch.no_grad():
                        self._embedding.weight[vector_ind] = \
                            flat_input[rand_batch]
                        self.usage_counts[vector_ind] = 100

        # convert quantized from BHWC -> BCHW
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        encodings = encodings.reshape((b, h, w, self._num_embeddings))
        return [quantized, loss, perplexity, encodings]


# Discrete Latent Spaces
class VQLatentSpace1D(nn.Module):
    """VQ_Latent_space Implementation
    Taken from https://github.com/zalandoresearch/pytorch-vq-vae
    Adapted to include a solution to codebook collapse. See: 
    https://www.reddit.com/r/MachineLearning/comments/nxjqvb/d_preventing_index_collapse_in_vqvae/h1fu9cs/
    Also from openai paper: https://arxiv.org/pdf/2005.00341.pdf
    >   VQ-VAEs are known to suffer from codebook collapse, wherein all encodings get mapped to a 
        single or few embedding vectors while the other embedding vectors in the codebook are not
        used, reducing the information capacity of the bottleneck. To prevent this, we use random
        restarts: when the mean usage of a codebook vector falls below a threshold, we randomly 
        reset it to one of the encoder outputs from the current batch
    """ # noqa
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VQLatentSpace1D, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(
            self._num_embeddings, 
            self._embedding_dim
        )
        self._embedding.weight.data \
            .uniform_(
                -1/self._num_embeddings, 
                1/self._num_embeddings)
        self._commitment_cost = commitment_cost

        self.usage_counts = {i: 100 for i in \
            range(self._num_embeddings)}

    def forward(self, inputs, training=True):
        # convert inputs from BCHW -> BHWC
        inputs = rearrange(inputs, 'b c w -> b w c') \
            .contiguous()
        b, w, c = inputs.shape
        input_shape = (b, w, c)

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True) 
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(
                flat_input, 
                self._embedding.weight.t()
            )
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1) \
            .unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], 
            self._num_embeddings, 
            device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        if training:
            for ind in range(self._num_embeddings):
                if ind not in encoding_indices.t().tolist()[0]:
                    self.usage_counts[ind] -= 1
                else:
                    self.usage_counts[ind] += 1

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight) \
            .view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        if training:
            for vector_ind, count in self.usage_counts.items():
                if count == 0:
                    rand_batch = np.random.randint(0, flat_input.shape[0])
                    with torch.no_grad():
                        self._embedding.weight[vector_ind] = \
                            flat_input[rand_batch]
                        self.usage_counts[vector_ind] = 100

        # convert quantized from BWC -> BCW
        quantized = quantized.permute(0, 2, 1).contiguous()
        encodings = encodings.reshape((b, w, self._num_embeddings))
        return [quantized, loss, perplexity, encodings]
