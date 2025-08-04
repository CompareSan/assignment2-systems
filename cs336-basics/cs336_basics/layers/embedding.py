import torch.nn as nn
import torch


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor: 
        return self.weight[token_ids]
    











