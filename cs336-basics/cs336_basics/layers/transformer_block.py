import torch
import torch.nn as nn
from cs336_basics.layers.multi_head_self_attention import MultiHeadSelfAttentionWithRoPE
from cs336_basics.layers.rms_norm import RMSNorm
from cs336_basics.layers.swiglu_ff import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int):
        super().__init__()

        self.rmsnorm1 = RMSNorm(d_model)
        self.mha = MultiHeadSelfAttentionWithRoPE(d_model, num_heads, theta, max_seq_len)
        self.swiglu = SwiGLU(d_model, d_ff)
        self.rmsnorm2 = RMSNorm(d_model)

    def forward(self, x: torch.Tensor):
        token_pos = torch.arange(x.shape[1], device=x.device, dtype=torch.long)
        y = x + self.mha(self.rmsnorm1(x), token_pos)
        return y + self.swiglu(self.rmsnorm2(y))


