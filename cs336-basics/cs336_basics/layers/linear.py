import torch.nn as nn
import torch


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the linear layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        return x @ self.weight.t()
        



