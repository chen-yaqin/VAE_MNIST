import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    
    def __init__(self, num_embeddings=512, embedding_dim=64, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z_e):
        batch_size, latent_dim = z_e.shape

        distances = torch.cdist(z_e, self.embedding.weight)  # (batch_size, num_embeddings)

        encoding_indices = torch.argmin(distances, dim=-1)  # (batch_size,)

        quantized = self.embedding(encoding_indices)  # (batch_size, latent_dim)

        vq_loss = self.commitment_cost * F.mse_loss(z_e.detach(), quantized) + \
                  F.mse_loss(z_e, quantized.detach())

        quantized = z_e + (quantized - z_e).detach()

        return quantized, vq_loss

class VectorQuantizerEMA(nn.Module):

    def __init__(self, num_embeddings=512, embedding_dim=64, commitment_cost=0.25, decay=0.99):
        super(VectorQuantizerEMA, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

        self.ema_cluster_size = torch.zeros(num_embeddings)
        self.ema_w = self.embedding.weight.clone()

    def forward(self, z_e):

        batch_size, latent_dim = z_e.shape

        distances = torch.cdist(z_e, self.embedding.weight)  # (batch_size, num_embeddings)

        encoding_indices = torch.argmin(distances, dim=-1)  # (batch_size,)

        quantized = self.embedding(encoding_indices)  # (batch_size, latent_dim)

        vq_loss = self.commitment_cost * F.mse_loss(z_e.detach(), quantized) + \
                  F.mse_loss(z_e, quantized.detach())

        with torch.no_grad():
            encodings_one_hot = torch.zeros(batch_size, self.num_embeddings, device=z_e.device)
            encodings_one_hot.scatter_(1, encoding_indices.unsqueeze(1), 1)

            self.ema_cluster_size = self.decay * self.ema_cluster_size + (1 - self.decay) * torch.sum(encodings_one_hot, dim=0)

            self.ema_w = self.decay * self.ema_w + (1 - self.decay) * torch.matmul(encodings_one_hot.t(), z_e)

            n = torch.sum(self.ema_cluster_size)
            self.ema_cluster_size = (self.ema_cluster_size + 1e-5) / (n + 1e-5) * n
            self.embedding.weight.data = self.ema_w / self.ema_cluster_size.unsqueeze(1)

        quantized = z_e + (quantized - z_e).detach()

        return quantized, vq_loss