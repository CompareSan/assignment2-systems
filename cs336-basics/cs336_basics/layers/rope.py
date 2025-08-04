import torch
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        inv_freq = 1.0 / (
            theta
            **(torch.arange(0, d_k, 2, device=device)[: (d_k // 2)].float() / d_k)
        )

        i = torch.arange(max_seq_len, device=device, dtype=torch.float)
        i_div_inv_freq_matrix = i.unsqueeze(-1)@inv_freq.unsqueeze(0)
        cache = torch.stack([i_div_inv_freq_matrix.cos(), i_div_inv_freq_matrix.sin()], dim=-1)
        self.register_buffer("cache", cache, persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        rope_cache = self.cache[token_positions]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)


        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )
        x_out = x_out.reshape(*x.shape[:-1], -1)
        return x_out
