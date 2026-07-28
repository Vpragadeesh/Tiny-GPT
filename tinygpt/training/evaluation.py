"""Loss estimation helpers."""

import torch


@torch.no_grad()
def estimate_loss(model, get_batch, eval_iters, use_activation_checkpoint=False):
    """Estimate train/val loss over eval_iters batches."""
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y, use_activation_checkpoint=use_activation_checkpoint)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out
