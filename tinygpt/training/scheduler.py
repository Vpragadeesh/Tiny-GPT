"""Learning rate scheduler: linear warmup → cosine decay."""

import math


def get_lr(step, lr, warmup_steps, max_iters):
    """Linear warmup → cosine decay to 10% of peak LR."""
    if step < warmup_steps:
        return lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, max_iters - warmup_steps)
    return lr * 0.1 + 0.5 * lr * 0.9 * (1 + math.cos(math.pi * progress))
