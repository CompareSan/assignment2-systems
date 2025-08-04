import torch
import math


def cosine_annealing_scheduler(t: int, a_max: float, a_min: float, t_warmup: int, t_post: int) -> float:
    if t < t_warmup:
        return t/t_warmup * a_max

    if t_warmup <= t < t_post:
        return a_min + 0.5 * (a_max - a_min) * (1 + math.cos(math.pi * (t - t_warmup) / (t_post - t_warmup)))

    if t >= t_post:
        return a_min