"""Checkpoint save/load helpers."""

import os
import torch
import gc


def detect_format(opt_state):
    if "master" in opt_state and "m" in opt_state and "v" in opt_state:
        return "offload"
    return "torch"


def convert_optimizer_state(opt_state, target_format, model_params):
    current = detect_format(opt_state)
    if current == target_format:
        return opt_state

    if target_format == "torch" and current == "offload":
        state = {}
        for i in range(len(opt_state["m"])):
            state[i] = {
                "step": torch.tensor(float(opt_state["t"])),
                "exp_avg": opt_state["m"][i],
                "exp_avg_sq": opt_state["v"][i],
            }
        param_groups = [{"params": list(range(len(opt_state["m"])))}]
        return {"state": state, "param_groups": param_groups}

    if target_format == "offload" and current == "torch":
        t = 0
        m = []
        v = []
        master = [p.data.float().cpu() for p in model_params]

        if "state" in opt_state:
            for i in range(len(model_params)):
                if i in opt_state["state"]:
                    m.append(opt_state["state"][i]["exp_avg"].float().cpu())
                    v.append(opt_state["state"][i]["exp_avg_sq"].float().cpu())
                    t = int(opt_state["state"][i].get("step", t))
                else:
                    m.append(torch.zeros_like(master[i]))
                    v.append(torch.zeros_like(master[i]))
        return {"t": t, "master": master, "m": m, "v": v}

    return opt_state


def save_checkpoint(step, model, optimizer, train_loss, val_loss, path,
                    attention_type=None):
    """Save model + optimizer + training state to disk with atomic write."""
    state = {
        "step":       step,
        "model_state": model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss":   val_loss,
    }
    if attention_type is not None:
        state["attention_type"] = attention_type
        
    tmp_path = path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def load_model_weights(path, model, attention_type=None):
    """Load only model weights, avoiding optimizer state memory spike."""
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        
    sd = ckpt["model_state"]

    # Filter out layers that don't exist in current model (handles layer count changes)
    model_keys = set(model.state_dict().keys())
    sd = {k: v for k, v in sd.items() if k in model_keys}

    if attention_type is not None:
        ckpt_attn = ckpt.get("attention_type", "softmax")
        if ckpt_attn != attention_type:
            print(f"  [warn] checkpoint attention={ckpt_attn}, "
                  f"running with {attention_type} — skipping attention weights")
            sd = {k: v for k, v in sd.items() if ".attn." not in k}
        model.load_state_dict(sd, strict=(ckpt_attn == attention_type))
    else:
        model.load_state_dict(sd, strict=False)
        
    step = ckpt.get('step', 0)
    train_loss = ckpt.get('train_loss', 0.0)
    val_loss = ckpt.get('val_loss', 0.0)
    
    del ckpt
    gc.collect()
    
    return step, train_loss, val_loss


def load_checkpoint(path, model, optimizer, attention_type=None):
    """Load checkpoint and return (step, val_loss)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    val_loss = ckpt.get("val_loss", 0.0)
    train_loss = ckpt.get("train_loss", 0.0)
    if val_loss != val_loss or train_loss != train_loss:
        del ckpt
        gc.collect()
        raise ValueError("NaN checkpoint detected")
        
    sd = ckpt["model_state"]

    if attention_type is not None:
        ckpt_attn = ckpt.get("attention_type", "softmax")
        if ckpt_attn != attention_type:
            print(f"  [warn] checkpoint attention={ckpt_attn}, "
                  f"running with {attention_type} — skipping attention weights")
            sd = {k: v for k, v in sd.items() if ".attn." not in k}
        model.load_state_dict(sd, strict=(ckpt_attn == attention_type))
    else:
        model.load_state_dict(sd)

    opt_sd = ckpt["optimizer"]
    target_format = "offload" if hasattr(optimizer, "master") else "torch"
    opt_sd = convert_optimizer_state(opt_sd, target_format, list(model.parameters()))
    optimizer.load_state_dict(opt_sd)

    step = ckpt['step']
    val_loss = ckpt['val_loss']
    
    print(f"  Resumed from step {step}  "
          f"(train {ckpt.get('train_loss', 0.0):.4f}, val {val_loss:.4f})")
          
    del ckpt
    gc.collect()
    return step, val_loss
