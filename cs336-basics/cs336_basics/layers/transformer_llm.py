import torch
import torch.nn as nn
from cs336_basics.layers.embedding import Embedding
from cs336_basics.layers.transformer_block import TransformerBlock
from cs336_basics.layers.rms_norm import RMSNorm
from cs336_basics.layers.linear import Linear


class Transformer(nn.Module):
    def __init__(self, vocab_size: int, context_len: int, num_layers: int,
                 d_model: int, num_heads: int, d_ff: int, theta: float,):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, theta, context_len) for _ in range(num_layers)]) 
        self.rmsnorm = RMSNorm(d_model)
        self.linear = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        x = self.rmsnorm(x)
        return self.linear(x)





