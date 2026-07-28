"""Checkpoint save/load helpers."""

import torch


def save_checkpoint(step, model, optimizer, train_loss, val_loss, path,
                    attention_type=None):
    """Save model + optimizer + training state to disk."""
    state = {
        "step":       step,
        "model_state": model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss":   val_loss,
    }
    if attention_type is not None:
        state["attention_type"] = attention_type
    torch.save(state, path)


def load_checkpoint(path, model, optimizer, attention_type=None):
    """Load checkpoint and return (step, val_loss)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state"]

    # Handle weight tying: remove separate head.weight if present
    if 'head.weight' in sd:
        del sd['head.weight']
    # Also handle any 'model.head.weight' key
    sd = {k: v for k, v in sd.items() if k != 'model.head.weight'}

    if attention_type is not None:
        ckpt_attn = ckpt.get("attention_type", "softmax")
        if ckpt_attn != attention_type:
            print(f"  [warn] checkpoint attention={ckpt_attn}, "
                  f"running with {attention_type} — skipping attention weights")
            sd = {k: v for k, v in sd.items() if ".attn." not in k}
        model.load_state_dict(sd, strict=(ckpt_attn == attention_type))
    else:
        model.load_state_dict(sd)

    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"  Resumed from step {ckpt['step']}  "
          f"(train {ckpt['train_loss']:.4f}, val {ckpt['val_loss']:.4f})")
    return ckpt["step"], ckpt["val_loss"]
