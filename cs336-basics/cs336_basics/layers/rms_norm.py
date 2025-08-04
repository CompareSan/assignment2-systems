import torch.nn as nn
import torch



class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()

        self.eps = eps
        self.d_model = d_model
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor: #x (batch_size, seq_len, d_model)
        in_dtype = x.dtype
        x = x.to(torch.float32)
        result = x * (self.gain / (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt())
        return result.to(in_dtype)

