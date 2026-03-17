"""
MoE GPT – 0.5 Billion Parameter Language Model (DeepSpeed ZeRO-3)
==================================================================
Mixture-of-Experts GPT trained on FineWeb-Edu with DeepSpeed ZeRO-Infinity:
  - ZeRO Stage 3: All states partitioned across GPUs + CPU RAM offload
  - CPU Offloading: Parameters & optimizer states in CPU RAM
  - Memory efficient: Fits massive models on limited VRAM
  - Automatic gradient checkpointing & mixed precision (bfloat16)

Architecture
  12 Transformer layers  ×  (12-head attention  +  MoE FFN)
  8 expert FFNs per layer, top-2 routing
  Total params  ≈ 521 M   |   Active per token  ≈ 180 M

Run order:
    pip install torch tiktoken numpy datasets deepspeed
    python prepare_data.py          # once — downloads FineWeb-Edu
    deepspeed --num_gpus 1 main.py  # train with DeepSpeed
    python run.py                   # generate
"""

import os
import sys
import math
import gc
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
import tiktoken
import deepspeed
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
MICRO_BATCH   = 2                # samples per GPU forward pass (managed by DeepSpeed)
GRAD_ACCUM    = 8                # accumulate before optimizer step → eff. batch 16
EMBED_DIM     = 512              # model width
NUM_HEADS     = 8                # attention heads
NUM_LAYERS    = 8                # transformer blocks
NUM_EXPERTS   = 4                # expert FFNs per MoE layer
TOP_K         = 2                # experts activated per token
FFN_DIM       = EMBED_DIM * 4   # 2 048  (expert hidden dim)
DROPOUT       = 0.1
LR            = 1.5e-4           # peak learning rate
WARMUP_STEPS  = 500              # linear warmup
MAX_ITERS     = 30000            # total optimiser steps
EVAL_EVERY    = 100
EVAL_ITERS    = 50
AUX_LOSS_W    = 0.01             # load-balancing auxiliary loss weight
GRAD_CLIP     = 1.0
CHECKPOINT_DIR = "checkpoints"   # directory for saving checkpoints

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if DEVICE == "cuda" else torch.float32

print(f"Device          : {DEVICE.upper()}")
print(f"Precision       : {'BF16 (DeepSpeed)' if DTYPE == torch.bfloat16 else 'FP32'}")
print(f"Effective batch (default) : {MICRO_BATCH * GRAD_ACCUM}")
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
# 5. MODEL — Mixture-of-Experts GPT  (~0.5 B params)
# ═════════════════════════════════════════════════════════════════════════════

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with fused QKV projection."""

    def __init__(self):
        super().__init__()
        self.n_heads  = NUM_HEADS
        self.head_dim = EMBED_DIM // NUM_HEADS
        self.qkv      = nn.Linear(EMBED_DIM, 3 * EMBED_DIM, bias=False)
        self.proj      = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        self.attn_drop = nn.Dropout(DROPOUT)
        self.proj_drop = nn.Dropout(DROPOUT)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
                .view(1, 1, BLOCK_SIZE, BLOCK_SIZE),
        )

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)          # each (B, H, T, D)

        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att.float(), dim=-1).to(x.dtype)   # softmax in fp32
        att = self.attn_drop(att)

        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj_drop(self.proj(out))


class ExpertFFN(nn.Module):
    """Single expert: two-layer FFN with GELU."""

    def __init__(self):
        super().__init__()
        self.w1   = nn.Linear(EMBED_DIM, FFN_DIM)
        self.w2   = nn.Linear(FFN_DIM, EMBED_DIM)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        return self.drop(self.w2(self.act(self.w1(x))))


class MoELayer(nn.Module):
    """
    Mixture-of-Experts: routes each token to TOP_K of NUM_EXPERTS FFNs.
    Includes Switch-Transformer-style load-balancing auxiliary loss.
    """

    def __init__(self):
        super().__init__()
        self.router  = nn.Linear(EMBED_DIM, NUM_EXPERTS, bias=False)
        self.experts = nn.ModuleList([ExpertFFN() for _ in range(NUM_EXPERTS)])

    def forward(self, x):
        B, T, C = x.shape
        flat = x.reshape(-1, C)                              # (N, C)
        N = flat.shape[0]

        # ── routing ──
        logits = self.router(flat)                            # (N, E)
        probs  = F.softmax(logits.float(), dim=-1)            # fp32 for stability

        top_w, top_i = torch.topk(probs, TOP_K, dim=-1)      # (N, K)
        top_w = (top_w / top_w.sum(dim=-1, keepdim=True)).to(x.dtype)

        # ── load-balancing loss ──
        one_hot = F.one_hot(top_i, NUM_EXPERTS).float().sum(dim=1)   # (N, E)
        f = one_hot.mean(dim=0)
        P = probs.mean(dim=0)
        aux_loss = NUM_EXPERTS * (f * P).sum()

        # ── dispatch to experts ──
        out = torch.zeros_like(flat)
        for i, expert in enumerate(self.experts):
            mask = (top_i == i).any(dim=-1)                   # (N,)
            if not mask.any():
                continue
            tokens  = flat[mask]                              # (n_i, C)
            e_out   = expert(tokens)                          # (n_i, C)
            match   = (top_i[mask] == i).to(x.dtype)          # (n_i, K)
            weights = (top_w[mask] * match).sum(-1, keepdim=True)
            out[mask] += weights * e_out

        return out.reshape(B, T, C), aux_loss


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: Attention + MoE, with residuals."""

    def __init__(self):
        super().__init__()
        self.ln1  = nn.LayerNorm(EMBED_DIM)
        self.attn = CausalSelfAttention()
        self.ln2  = nn.LayerNorm(EMBED_DIM)
        self.moe  = MoELayer()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        moe_out, aux = self.moe(self.ln2(x))
        x = x + moe_out
        return x, aux


class MoEGPT(nn.Module):
    """
    Full MoE-GPT model (~521 M parameters, ~180 M active per token).

    1. Token + positional embeddings
    2. 12 × Transformer blocks  (self-attention + MoE FFN)
    3. Final layer-norm → linear head (weight-tied with token embedding)
    """

    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, EMBED_DIM)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, EMBED_DIM)
        self.drop    = nn.Dropout(DROPOUT)
        self.blocks  = nn.ModuleList([TransformerBlock() for _ in range(NUM_LAYERS)])
        self.ln_f    = nn.LayerNorm(EMBED_DIM)
        self.head    = nn.Linear(EMBED_DIM, vocab_size, bias=False)

        # Weight tying saves ~38 M params and improves training
        self.head.weight = self.tok_emb.weight
        self._init_weights()

    def _init_weights(self):
        """GPT-2-style init with scaled residual projections."""
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif "bias" in name:
                nn.init.zeros_(p)
        scale = (2 * NUM_LAYERS) ** -0.5
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=0.02 * scale)
            for expert in block.moe.experts:
                nn.init.normal_(expert.w2.weight, mean=0.0, std=0.02 * scale)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.drop(
            self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        )

        total_aux = 0.0
        for block in self.blocks:
            if self.training:
                x, aux = grad_checkpoint(block, x, use_reentrant=False)
            else:
                x, aux = block(x)
            total_aux = total_aux + aux

        logits = self.head(self.ln_f(x))

        loss = None
        if targets is not None:
            ce   = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
            loss = ce + AUX_LOSS_W * total_aux
        return logits, loss

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens=200, temperature=0.8):
        self.eval()
        ids = encode(prompt)
        idx = torch.tensor([ids], dtype=torch.long, device=DEVICE)

        for _ in range(max_new_tokens):
            ctx = idx[:, -BLOCK_SIZE:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :].float() / temperature
            probs  = F.softmax(logits, dim=-1)
            nxt    = torch.multinomial(probs, 1)
            idx    = torch.cat([idx, nxt], dim=1)

        self.train()
        return decode(idx[0].tolist())

# ═════════════════════════════════════════════════════════════════════════════
# 6. CHECKPOINT HELPERS
# ═════════════════════════════════════════════════════════════════════════════

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save_checkpoint(step, model, train_loss, val_loss, path):
    """Save model and training state to disk."""
    # DeepSpeed handles checkpointing, but we also save basic metadata
    checkpoint = {
        "step": step,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "model_state": model.state_dict() if hasattr(model, 'state_dict') else None,
    }
    torch.save(checkpoint, path)

def load_checkpoint(path, model):
    """Load checkpoint and return the step to resume from."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("model_state"):
        model.load_state_dict(ckpt["model_state"])
    print(f"  Resumed from step {ckpt['step']}  "
          f"(train {ckpt['train_loss']:.4f}, val {ckpt['val_loss']:.4f})")
    return ckpt["step"], ckpt["val_loss"]

# ═════════════════════════════════════════════════════════════════════════════
# 7. LEARNING-RATE SCHEDULE  (linear warmup → cosine decay to 10 %)
# ═════════════════════════════════════════════════════════════════════════════

def get_lr(step):
    if step < WARMUP_STEPS:
        return LR * step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, MAX_ITERS - WARMUP_STEPS)
    return LR * 0.1 + 0.5 * LR * 0.9 * (1 + math.cos(math.pi * progress))


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
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=(DTYPE == torch.bfloat16)):
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
    model = MoEGPT()
    n_total  = sum(p.numel() for p in model.parameters())
    _expert1 = sum(p.numel() for p in model.blocks[0].moe.experts[0].parameters())
    n_active = n_total - _expert1 * (NUM_EXPERTS - TOP_K) * NUM_LAYERS

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
        console=console,
        refresh_per_second=4,
    ) as progress:
        total_steps = MAX_ITERS - start_step
        task = progress.add_task(
            "Training", total=total_steps,
            train_loss="--.----", val_loss="--.----", lr="--.------", end_clock="--:--",
        )

        step = start_step
        micro_loss_sum = 0.0
        micro_loss_count = 0
        total_micro_steps = MAX_ITERS * GRAD_ACCUM

        for micro_step in range(start_step * GRAD_ACCUM + 1, total_micro_steps + 1):

            lr = get_lr(step + 1)
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

            progress.update(
                task,
                advance=1,
                train_loss=f"{accum_loss:.4f}",
                lr=f"{lr:.6f}",
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
                    end_clock=get_eta_clock(progress, task),
                )
                progress.console.print(
                    f"  [bold]Step {step:>5}[/]  │  "
                    f"[yellow]Train {losses['train']:.4f}[/]  │  "
                    f"[cyan]Val {losses['val']:.4f} ({trend}, Δ {delta:+.4f})[/]  │  "
                    f"[magenta]LR {lr:.6f}[/]"
                )

                # Save checkpoints
                save_checkpoint(
                    step, model_engine.module if hasattr(model_engine, 'module') else model_engine,
                    losses["train"], losses["val"],
                    os.path.join(CHECKPOINT_DIR, "latest.pt"),
                )
                if losses["val"] < best_val:
                    best_val = losses["val"]
                    save_checkpoint(
                        step, model_engine.module if hasattr(model_engine, 'module') else model_engine,
                        losses["train"], losses["val"],
                        os.path.join(CHECKPOINT_DIR, "best.pt"),
                    )
                    progress.console.print(
                        f"  [bold green]★ New best val loss: {best_val:.4f}  (saved best.pt)[/]"
                    )

            if step >= MAX_ITERS:
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
        output = model_eval.generate(prompt, max_new_tokens=120, temperature=0.7)
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
        output = model_eval.generate(prompt, max_new_tokens=150, temperature=0.8)
        print(f"\n{output.strip()}")

    print("\nGoodbye!")
