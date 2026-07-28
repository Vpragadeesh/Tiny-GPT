"""
Tiny-GPT – Dense Transformer Language Model (DeepSpeed ZeRO-2)
==============================================================
Dense GPT trained on FineWeb-Edu with DeepSpeed ZeRO-2:
  - ZeRO Stage 2: Optimizer states partitioned across GPUs + CPU offload
  - BF16 mixed precision with FlashAttention
  - Automatic gradient checkpointing support

Architecture
  8 Transformer layers  ×  (8-head attention  +  Dense FFN)
  Total params  ≈ 101 M

Run order:
    pip install torch tiktoken numpy datasets deepspeed
    python prepare_data.py              # once — downloads FineWeb-Edu
    deepspeed --num_gpus 1 main_deepspeed.py  # train with DeepSpeed
    python run.py                       # generate
"""

import os
import sys
import math
import gc
import json
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
import tiktoken
import deepspeed
from model import CausalSelfAttention, FeedForward, TransformerBlock, TinyGPT
from tinygpt.training import get_lr, estimate_loss
from rich.progress import (
    Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn,
    SpinnerColumn, MofNCompleteColumn,
)
from rich.console import Console

console = Console()
IST = ZoneInfo("Asia/Kolkata")

# ═════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA  (memory-mapped .bin files from prepare_data.py)
# ═════════════════════════════════════════════════════════════════════════════

DATA_DIR = "data"
for split in ("train", "val", "test"):
    path = os.path.join(DATA_DIR, f"{split}.bin")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n[ERROR] '{path}' not found.\n"
            "Run  python prepare_data.py  first."
        )

train_data = np.memmap(os.path.join(DATA_DIR, "train.bin"), dtype=np.uint16, mode="r")
val_data   = np.memmap(os.path.join(DATA_DIR, "val.bin"),   dtype=np.uint16, mode="r")
test_data  = np.memmap(os.path.join(DATA_DIR, "test.bin"),  dtype=np.uint16, mode="r")

print("Dataset loaded (memory-mapped)")
print(f"  Train : {len(train_data):>12,} tokens")
print(f"  Val   : {len(val_data):>12,} tokens")
print(f"  Test  : {len(test_data):>12,} tokens")
print()

# ═════════════════════════════════════════════════════════════════════════════
# 2. TOKENISER – GPT-2 BPE  (matches prepare_data.py)
# ═════════════════════════════════════════════════════════════════════════════

enc        = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab                      # 50 257

def encode(text: str) -> list:
    return enc.encode_ordinary(text)

def decode(ids: list) -> str:
    return enc.decode(ids)

print(f"Tokeniser : GPT-2 BPE  (vocab {vocab_size:,})")
print()

# ═════════════════════════════════════════════════════════════════════════════
# 3. HYPERPARAMETERS
# ═════════════════════════════════════════════════════════════════════════════

BLOCK_SIZE    = 64               # context window (tokens)
MICRO_BATCH   = 8                # samples per GPU forward pass (managed by DeepSpeed)
GRAD_ACCUM    = 4                # accumulate before optimizer step → eff. batch 16
EMBED_DIM     = 512              # model width
NUM_HEADS     = 8                # attention heads
NUM_LAYERS    = 8                # transformer blocks
FFN_DIM       = EMBED_DIM * 4   # 2 048  (expert hidden dim)
DROPOUT       = 0.1
LR            = 5.0e-5           # peak learning rate (tuned for 101M model + batch 16)
WARMUP_STEPS  = 500              # linear warmup
MAX_ITERS     = 120_000           # total optimiser steps
EVAL_EVERY    = 2_000
EVAL_ITERS    = 50
GRAD_CLIP     = 1.0
CHECKPOINT_DIR = "checkpoints"   # directory for saving checkpoints
# ── Auto-stop ───────────────────────────────────────────────────────────
PATIENCE            = 5          # eval steps without improvement → stop
LOSS_EXPLODE_FACTOR = 1.5        # stop if val loss > best * factor
NAN_STOP            = True       # stop immediately on NaN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if DEVICE == "cuda" else torch.float32

ALLOW_TF32 = os.environ.get("ALLOW_TF32", "1") == "1"
USE_TORCH_COMPILE = os.environ.get("USE_TORCH_COMPILE", "0") == "1"
USE_ACTIVATION_CHECKPOINT = os.environ.get("USE_ACTIVATION_CHECKPOINT", "1") == "1"

if DEVICE == "cuda":
    # Throughput-oriented CUDA settings.
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = ALLOW_TF32
    torch.backends.cudnn.allow_tf32 = ALLOW_TF32
    torch.set_float32_matmul_precision("high")

print(f"Device          : {DEVICE.upper()}")
print(f"Precision       : {'BF16 (DeepSpeed)' if DTYPE == torch.bfloat16 else 'FP32'}")
print(f"Eff batch (Py) : {MICRO_BATCH * GRAD_ACCUM}  (DeepSpeed config may override)")
print(f"TF32            : {'ON' if (DEVICE == 'cuda' and ALLOW_TF32) else 'OFF'}")
print(f"Torch compile   : {'ON' if USE_TORCH_COMPILE else 'OFF'}")
print(f"Act checkpoint  : {'ON' if USE_ACTIVATION_CHECKPOINT else 'OFF'}")
print()

# ═════════════════════════════════════════════════════════════════════════════
# 4. DATA LOADER
# ═════════════════════════════════════════════════════════════════════════════

def get_batch(split="train"):
    data = {"train": train_data, "val": val_data, "test": test_data}[split]
    ix = np.random.randint(0, len(data) - BLOCK_SIZE, size=(MICRO_BATCH,))
    x = np.stack([data[i   : i + BLOCK_SIZE    ].astype(np.int64) for i in ix])
    y = np.stack([data[i+1 : i + BLOCK_SIZE + 1].astype(np.int64) for i in ix])
    return torch.from_numpy(x).to(DEVICE), torch.from_numpy(y).to(DEVICE)

# ═════════════════════════════════════════════════════════════════════════════
# 5. MODEL — see model.py for TinyGPT, TransformerBlock, FeedForward, CausalSelfAttention
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate(model, prompt: str, max_new_tokens=200, temperature=0.8, top_k=50, top_p=0.9):
    model.eval()
    ids = encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=DEVICE)

    for _ in range(max_new_tokens):
        ctx = idx[:, -BLOCK_SIZE:]
        logits, _ = model(ctx)
        logits = logits[:, -1, :].float() / temperature

        if top_k is not None:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float("-inf")

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumsum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumsum_probs > top_p
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[:, indices_to_remove] = float("-inf")

        probs  = F.softmax(logits, dim=-1)
        nxt    = torch.multinomial(probs, 1)
        idx    = torch.cat([idx, nxt], dim=1)

    model.train()
    return decode(idx[0].tolist())

# ═════════════════════════════════════════════════════════════════════════════
# 6. CHECKPOINT HELPERS
# ═════════════════════════════════════════════════════════════════════════════

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def _strip_orig_mod_prefix(state_dict):
    out = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            out[k[len("_orig_mod."):]] = v
        else:
            out[k] = v
    return out


def _add_orig_mod_prefix(state_dict):
    out = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            out[k] = v
        else:
            out[f"_orig_mod.{k}"] = v
    return out


def _align_state_dict_for_model(state_dict, model):
    """Align checkpoint keys with model keys (compiled vs non-compiled)."""
    model_keys = list(model.state_dict().keys())
    if not model_keys:
        return state_dict

    model_has_orig = model_keys[0].startswith("_orig_mod.")
    ckpt_keys = list(state_dict.keys())
    ckpt_has_orig = bool(ckpt_keys) and ckpt_keys[0].startswith("_orig_mod.")

    if model_has_orig and not ckpt_has_orig:
        return _add_orig_mod_prefix(state_dict)
    if not model_has_orig and ckpt_has_orig:
        return _strip_orig_mod_prefix(state_dict)
    return state_dict

def save_checkpoint(step, model, train_loss, val_loss, path):
    """Save model and training state to disk."""
    # DeepSpeed handles checkpointing, but we also save basic metadata
    model_state = model.state_dict() if hasattr(model, "state_dict") else None
    if model_state is not None:
        # Store canonical keys so checkpoints are reusable across compile modes.
        model_state = _strip_orig_mod_prefix(model_state)

    checkpoint = {
        "step": step,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "model_state": model_state,
    }
    torch.save(checkpoint, path)

def load_checkpoint(path, model):
    """Load checkpoint and return the step to resume from."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("model_state"):
        model_state = _align_state_dict_for_model(ckpt["model_state"], model)
        model.load_state_dict(model_state)
    print(f"  Resumed from step {ckpt['step']}  "
          f"(train {ckpt['train_loss']:.4f}, val {ckpt['val_loss']:.4f})")
    return ckpt["step"], ckpt["val_loss"]

# ═════════════════════════════════════════════════════════════════════════════
# 7. LEARNING-RATE SCHEDULE — see tinygpt.training.get_lr
# ═════════════════════════════════════════════════════════════════════════════


def get_eta_clock(progress, task_id):
    """Return estimated finish time in IST as HH:MM."""
    remaining = progress.tasks[task_id].time_remaining
    if remaining is None:
        return "--:--"
    end_at = datetime.now(IST) + timedelta(seconds=max(0.0, remaining))
    return end_at.strftime("%H:%M")

# ═════════════════════════════════════════════════════════════════════════════
# 8. LOSS ESTIMATION
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(EVAL_ITERS):
            x, y = get_batch(split)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda" and DTYPE == torch.bfloat16)):
                _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out

# ═════════════════════════════════════════════════════════════════════════════
# 9. DEEPSPEED INITIALIZATION & TRAINING
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Clear cache
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # Initialize model
    model = TinyGPT(vocab_size, BLOCK_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS,
                    FFN_DIM, DROPOUT)
    if DEVICE == "cuda" and USE_TORCH_COMPILE:
        model = torch.compile(model, mode="max-autotune", fullgraph=False)

    n_total  = sum(p.numel() for p in model.parameters())
    n_active = n_total  # In a dense model, all parameters are active

    print(f"Total  parameters : {n_total:>14,}")
    print(f"Active per token  : {n_active:>14,}")
    print()

    # Ensure LOCAL_RANK is set so DeepSpeed's sanity checks pass when running
    # this script directly with `python main_deepspeed.py` (single-GPU).
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"

    # Initialize torch.distributed if not already initialized (single-process setup).
    if not torch.distributed.is_initialized():
        init_kwargs = {
            "backend": "nccl" if DEVICE == "cuda" else "gloo",
            "init_method": "tcp://127.0.0.1:29500",
            "rank": 0,
            "world_size": 1,
        }
        if DEVICE == "cuda":
            init_kwargs["device_id"] = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        torch.distributed.init_process_group(**init_kwargs)

    # Load DeepSpeed config (launcher may provide a mode-specific config path).
    ds_config_path = os.environ.get("DS_CONFIG_PATH", "ds_config.json")
    with open(ds_config_path) as f:
        ds_config = json.load(f)

    # Keep runtime batch settings aligned with DeepSpeed config.
    MICRO_BATCH = int(ds_config.get("train_micro_batch_size_per_gpu", MICRO_BATCH))
    GRAD_ACCUM = int(ds_config.get("gradient_accumulation_steps", GRAD_ACCUM))

    print(f"DeepSpeed micro-batch : {MICRO_BATCH}")
    print(f"DeepSpeed grad accum  : {GRAD_ACCUM}")
    print(f"DeepSpeed eff batch   : {MICRO_BATCH * GRAD_ACCUM}")
    print()

    # Initialize DeepSpeed engine
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        args=type("args", (), {"local_rank": int(os.environ.get("LOCAL_RANK", 0))})(),
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
        dist_init_required=False,
    )

    print(f"[DeepSpeed] Initialized with ZeRO Stage {ds_config['zero_optimization']['stage']}")
    print(f"[DeepSpeed] Device: {model_engine.device}")
    print()

    # Training state
    start_step = 0
    best_val = float("inf")
    prev_val = None
    steps_without_improvement = 0
    stop_reason = None

    # Auto-resume from latest checkpoint (with NaN guard)
    latest_ckpt = os.path.join(CHECKPOINT_DIR, "latest.pt")
    if os.path.exists(latest_ckpt):
        try:
            _c = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
            if _c.get("val_loss") != _c.get("val_loss") or _c.get("train_loss") != _c.get("train_loss"):
                print("Checkpoint has NaN losses — deleting and starting fresh")
                os.remove(latest_ckpt)
            else:
                print("Checkpoint found — resuming …")
                start_step, best_val = load_checkpoint(latest_ckpt, model)
                print()
        except Exception as e:
            print(f"Checkpoint corrupted ({e}) — starting fresh")
    else:
        print("No checkpoint found — starting fresh training")
        print()

    # ─────────────────────────────────────────────────────────────────────────
    # TRAINING LOOP
    # ─────────────────────────────────────────────────────────────────────────

    console.rule("[bold green]Training started (DeepSpeed)")
    print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        TextColumn("•"),
        TextColumn("[green]ETA {task.fields[end_clock]} IST"),
        TextColumn("•"),
        TextColumn("[yellow]loss {task.fields[train_loss]}"),
        TextColumn("[cyan]val {task.fields[val_loss]}"),
        TextColumn("[magenta]lr {task.fields[lr]}"),
        TextColumn("•"),
        TextColumn("[bold cyan]{task.fields[tok_s]} tok/s"),
        console=console,
        refresh_per_second=4,
    ) as progress:
        total_steps = MAX_ITERS - start_step
        task = progress.add_task(
            "Training", total=total_steps,
            train_loss="--.----", val_loss="--.----", lr="--.------", end_clock="--:--",
            tok_s="------",
        )

        step = start_step
        micro_loss_sum = 0.0
        micro_loss_count = 0
        total_micro_steps = MAX_ITERS * GRAD_ACCUM
        step_start_time = time.perf_counter()
        tokens_per_step = MICRO_BATCH * GRAD_ACCUM * BLOCK_SIZE

        for micro_step in range(start_step * GRAD_ACCUM + 1, total_micro_steps + 1):

            lr = get_lr(step + 1, LR, WARMUP_STEPS, MAX_ITERS)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            x, y = get_batch("train")
            _, loss = model_engine(x, y)

            model_engine.backward(loss)
            is_boundary = model_engine.is_gradient_accumulation_boundary()
            model_engine.step()

            micro_loss_sum += loss.item()
            micro_loss_count += 1

            if not is_boundary:
                continue

            step += 1
            accum_loss = micro_loss_sum / max(1, micro_loss_count)
            micro_loss_sum = 0.0
            micro_loss_count = 0

            now = time.perf_counter()
            elapsed = now - step_start_time
            step_start_time = now
            tok_s = tokens_per_step / elapsed if elapsed > 0 else 0

            progress.update(
                task,
                advance=1,
                train_loss=f"{accum_loss:.4f}",
                lr=f"{lr:.6f}",
                tok_s=f"{tok_s:,.0f}",
                end_clock=get_eta_clock(progress, task),
            )

            if step % EVAL_EVERY == 0:
                losses = estimate_loss(model_engine.module if hasattr(model_engine, "module") else model_engine)
                if prev_val is None:
                    trend = "init"
                    delta = 0.0
                else:
                    delta = losses["val"] - prev_val
                    if delta < -1e-6:
                        trend = "improving"
                    elif delta > 1e-6:
                        trend = "worse"
                    else:
                        trend = "flat"
                prev_val = losses["val"]

                progress.update(
                    task,
                    train_loss=f"{losses['train']:.4f}",
                    val_loss=f"{losses['val']:.4f}",
                    lr=f"{lr:.6f}",
                    tok_s=f"{tok_s:,.0f}",
                    end_clock=get_eta_clock(progress, task),
                )
                progress.console.print(
                    f"  [bold]Step {step:>5}[/]  │  "
                    f"[yellow]Train {losses['train']:.4f}[/]  │  "
                    f"[cyan]Val {losses['val']:.4f} ({trend}, Δ {delta:+.4f})[/]  │  "
                    f"[magenta]LR {lr:.6f}[/]  │  "
                    f"[bold cyan]{tok_s:,.0f} tok/s[/]"
                )

                # Save checkpoints
                save_checkpoint(
                    step, model_engine.module if hasattr(model_engine, 'module') else model_engine,
                    losses["train"], losses["val"],
                    os.path.join(CHECKPOINT_DIR, "latest.pt"),
                )
                if losses["val"] < best_val:
                    best_val = losses["val"]
                    steps_without_improvement = 0
                    save_checkpoint(
                        step, model_engine.module if hasattr(model_engine, 'module') else model_engine,
                        losses["train"], losses["val"],
                        os.path.join(CHECKPOINT_DIR, "best.pt"),
                    )
                    progress.console.print(
                        f"  [bold green]★ New best val loss: {best_val:.4f}  (saved best.pt)[/]"
                    )
                else:
                    steps_without_improvement += 1

                # ── Auto-stop checks ──
                val_l = losses["val"]
                if math.isnan(val_l) or math.isnan(losses["train"]):
                    stop_reason = "NaN loss detected"
                elif best_val > 0 and val_l > best_val * LOSS_EXPLODE_FACTOR:
                    stop_reason = f"Loss exploded: {val_l:.4f} > {best_val:.4f} × {LOSS_EXPLODE_FACTOR}"
                elif steps_without_improvement >= PATIENCE:
                    stop_reason = f"No improvement for {PATIENCE} evals (best {best_val:.4f})"

                if stop_reason:
                    progress.console.print(
                        f"  [bold red]⛔ STOPPING: {stop_reason}[/]"
                    )
                    save_checkpoint(
                        step, model_engine.module if hasattr(model_engine, 'module') else model_engine,
                        losses["train"], losses["val"],
                        os.path.join(CHECKPOINT_DIR, "latest.pt"),
                    )
                    break

            if step >= MAX_ITERS or stop_reason:
                break

    print()
    console.rule("[bold green]Training complete")
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # TEST EVALUATION
    # ─────────────────────────────────────────────────────────────────────────

    model_eval = model_engine.module if hasattr(model_engine, 'module') else model_engine
    model_eval.eval()
    test_losses = []
    with torch.no_grad():
        for _ in range(EVAL_ITERS):
            x, y = get_batch("test")
            _, loss = model_eval(x, y)
            test_losses.append(loss.item())
    test_loss = sum(test_losses) / len(test_losses)
    print(f"Test loss : {test_loss:.4f}")
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # TEXT GENERATION
    # ─────────────────────────────────────────────────────────────────────────

    prompts = [
        "The history of",
        "Scientists have discovered",
        "In the early twentieth century",
    ]

    print("=" * 60)
    print("Generated Text Samples")
    print("=" * 60)

    for prompt in prompts:
        output = generate(model_eval, prompt, max_new_tokens=120, temperature=0.7, top_k=50, top_p=0.9)
        print(f"\nPrompt : \"{prompt}\"")
        print(f"Output : {output.strip()}")
        print()

    # ─────────────────────────────────────────────────────────────────────────
    # INTERACTIVE MODE
    # ─────────────────────────────────────────────────────────────────────────

    print("=" * 60)
    print("Interactive Mode  (type 'quit' to exit)")
    print("=" * 60)

    while True:
        try:
            prompt = input("\nEnter a prompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt or prompt.lower() == "quit":
            break
        output = generate(model_eval, prompt, max_new_tokens=150, temperature=0.8, top_k=50, top_p=0.9)
        print(f"\n{output.strip()}")

    print("\nGoodbye!")
