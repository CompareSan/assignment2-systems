import torch 
from cs336_basics.layers.softmax  import softmax

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    # Q (batch_size, seq_len, d_k), K (batch_size, seq_len, d_k), V (batch_size, seq_len, d_v)
    # Q K.T (batch_size, seq_len, d_k)(batch_size, d_k, seq_len) -> scores (batch_size, seq_len, seq_len)
    # softmax(scores/scale) -> (batch_size, seq_len, seq_len)
    # softmax(scores/scale) V -> (batch_size, seq_len, seq_len) (batch_size, seq_len, d_v) -> (batch_size, seq_len, d_v)
    d_k = torch.tensor(Q.shape[-1])
    scores = Q @ K.transpose(-1, -2) / torch.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    return softmax(scores, dim=-1) @ V
