import torch


def log_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    logits = logits - torch.max(logits, dim=dim, keepdim=True).values
    return logits - torch.log(torch.sum(torch.exp(logits), dim=dim, keepdim=True))

def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor):
    # logits (b, s, vocab_size), # labels (b, s, )
    labels = labels.unsqueeze(-1) # (b, s, 1)
    log_probas = -log_softmax(logits, dim=-1)  # (b, s, vocab_size)
    # Gather the probabilities of the true label
    log_probas = torch.gather(log_probas, dim=-1, index=labels)  # (b, s, 1)
    log_probas = log_probas.squeeze(-1)  # (b, s)
    

    return torch.mean(log_probas)  # Average over all examples


    

