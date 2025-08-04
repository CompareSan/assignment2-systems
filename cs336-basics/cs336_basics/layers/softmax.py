import torch


def softmax(x: torch.Tensor, dim: int, t: float = 1.0) -> torch.Tensor: # x (batch, seq_len, d_model)
    x = x - torch.max(x, dim=dim, keepdim=True).values
    return torch.exp(x/t) / torch.sum(torch.exp(x/t), dim=dim, keepdim=True)