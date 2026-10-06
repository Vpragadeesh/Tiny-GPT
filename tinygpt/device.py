import os
import torch
from contextlib import nullcontext

def resolve_device():
    env_device = os.environ.get("TINYGPT_DEVICE")
    if env_device:
        device = "cpu" if env_device.lower() == "cpu" else env_device.lower()
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    if device == "cpu":
        threads = int(os.environ.get("OMP_NUM_THREADS", 2))
        torch.set_num_threads(threads)
        
    return device

def resolve_dtype(device):
    return torch.bfloat16 if device == "cuda" else torch.float32

def autocast_ctx(device, dtype):
    if device == "cuda":
        return torch.amp.autocast("cuda", dtype=dtype, enabled=(dtype == torch.bfloat16))
    else:
        # On CPU, fp32 is used, so no autocast is needed
        return nullcontext()
