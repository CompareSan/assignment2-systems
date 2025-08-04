from collections.abc import Callable, Iterable
import torch



class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999),
                 weight_decay: float = 1.0, eps: float = 10e-8):
        
        defaults = {"lr": lr,
                    "betas": betas,
                    "weight_decay": weight_decay,
                    "eps": eps}
        
        super().__init__(params, defaults)


    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                m = beta1*m + (1-beta1)*grad
                v = beta2*v + (1-beta2)*grad**2
                lr_adj = lr * (1 - beta2**(t + 1))**0.5 / (1 - beta1**(t + 1))

                p.data -= lr_adj * m / (v**0.5 + eps)

                p.data -= lr * weight_decay * p.data

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss
