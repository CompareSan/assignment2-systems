import torch
import torch.nn as nn
from cs336_basics.layers.linear import Linear
from cs336_basics.layers.rope import RoPE
from cs336_basics.layers.scaled_dot_product_attention import scaled_dot_product_attention



class MultiHeadSelfAttention(nn.Module): 
    def __init__(self, d_model: int, n_heads: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        # d_model = d_k / n_heads
        # d_k = d_q = d_v
        # x (batch_size, seq_len, d_model) - > (batch_size, seq_len, n_heads, d_k) -> (batch_size, n_heads, seq_len, d_k) 
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads #
        self.W_q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_k = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_v = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_o = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, n_heads, d_k)
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)   

        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device), diagonal=0)
        x = scaled_dot_product_attention(Q, K, V, mask) #(x (batch_size, n_heads, seq_len, d_k))
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model) #(batch_size, seq_len, n_heads, d_k))
    
        return self.W_o(x)  # (batch_size, seq_len, d_model)
class MultiHeadSelfAttentionWithRoPE(nn.Module): 
    def __init__(self, d_model: int, n_heads: int, theta: float, max_seq_len: int, 
                 device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        # d_model = d_k / n_heads
        # d_k = d_q = d_v
        # x (batch_size, seq_len, d_model) - > (batch_size, seq_len, n_heads, d_k) -> (batch_size, n_heads, seq_len, d_k) 
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads #
        self.W_q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_k = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_v = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_o = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = RoPE(theta, self.d_k, max_seq_len)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, n_heads, d_k)
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)   

        # Apply RoPE to Q and K
        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)

        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device), diagonal=0)
        x = scaled_dot_product_attention(Q, K, V, mask) #(x (batch_size, n_heads, seq_len, d_k))
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model) #(batch_size, seq_len, n_heads, d_k))
    
        return self.W_o(x)  # (batch_size, seq_len, d_model)