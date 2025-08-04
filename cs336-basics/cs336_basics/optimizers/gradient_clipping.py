import torch
from collections.abc import Iterable

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_norm: float, eps: float = 1e-6) -> None:
    """
    Clips the gradients of the parameters to prevent exploding gradients.

    Args:
        parameters (Iterable[torch.nn.Parameter]): Iterable of parameters whose gradients will be clipped.
        max_norm (float): Maximum allowed L2 norm of the gradients.
        eps (float): Small constant to prevent division by zero.
    """
    grads = [p.grad.detach().flatten() for p in parameters if p.grad is not None]
    if not grads:
        return

    all_grads = torch.cat(grads)
    total_norm = torch.norm(all_grads, p=2)

    if total_norm > max_norm:
        scale = max_norm / (total_norm + eps)
        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(scale)  # In-place scaling
