import torch.nn as nn
import torch
from cs336_basics.layers.linear import Linear


class SwiGLU(nn.Module):
    def __init__(self, d_model: int,
                d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,):
        super().__init__()

        self.linear1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.linear2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.linear3 = Linear(d_model, d_ff, device=device, dtype=dtype)



    def forward(self, x: torch.Tensor) -> torch.Tensor: # x (batch_size, seq_len, d_model)
        x = self.linear2((self.linear1(x) * torch.sigmoid(self.linear1(x))) * self.linear3(x)) #(batch_size, seq_len, d_ff) * (batch_size, seq_len, d_ff))
        return x # (batch_size, seq_len, d_model)