"""
Tiny-GPT – Dense Transformer Language Model (124M)
==================================================
GPT-2 Small architecture trained on FineWeb-Edu.
Fits on GPUs with as little as 4 GB VRAM via:
  - BF16 model weights on GPU
  - CPU-offloaded AdamW (optimizer states on RAM, not VRAM)
  - FlashAttention (O(N) fused attention)
  - Activation checkpointing (recompute activations to save VRAM)

Architecture
  12 Transformer layers  ×  (12-head attention  +  Dense FFN)
  Total params  ≈ 124 M

Run order:
    pip install torch tiktoken numpy datasets
    python prepare_data.py          # once — downloads FineWeb-Edu
    python main.py                  # train + generate
"""

import os
import math
import gc
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
os.environ.setdefault("TIKTOKEN_CACHE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiktoken_cache"))
import tiktoken
from rich.progress import (
    Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn,
    SpinnerColumn, MofNCompleteColumn,
)
from rich.console import Console
from rich.table import Table
from rich import print as rprint
from model import CausalSelfAttention, FeedForward, TransformerBlock, TinyGPT
from tinygpt.attention import LinearAttention
from tinygpt.training import CPUOffloadAdamW, make_optimizer, get_lr, save_checkpoint, load_checkpoint, estimate_loss
from tinygpt.device import resolve_device, resolve_dtype, autocast_ctx
from eval_suite import eval_suite

console = Console()

# ═════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA  (memory-mapped .bin files from prepare_data.py)
# ═════════════════════════════════════════════════════════════════════════════

DATA_DIR = "data"
train_data, val_data, test_data = None, None, None

def _load_datasets():
    global train_data, val_data, test_data
    if train_data is not None:
        return
        
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
# ═══════════════════════════════════════════════════════════════════════════════
# 3. HYPERPARAMETERS — 95M config for CPU (2c/8GB): 10L×768d×12H, batch 64
# ══════════════════════════════════════════════════════════════════════════════
BLOCK_SIZE    = 512              # context window (tokens)
MICRO_BATCH   = 4                # samples per forward pass (matches original)
GRAD_ACCUM    = 16               # gradient accumulation steps → eff. batch 64
                                 # Total tokens/step: MICRO_BATCH * GRAD_ACCUM * BLOCK_SIZE
EMBED_DIM     = 768              # model width
NUM_HEADS     = 12               # attention heads (768 / 12 = 64 head_dim)
NUM_LAYERS    = 10               # transformer blocks (~95M params)
FFN_DIM       = EMBED_DIM * 4   # 3 072
DROPOUT       = 0.1
LR            = 2.5e-4           # peak learning rate (higher for fewer layers)
WARMUP_STEPS  = 500              # linear warmup for stability
MAX_ITERS     = 50_000           # marathon training
EVAL_EVERY    = 500
EVAL_ITERS    = 50
USE_ACTIVATION_CHECKPOINT = True  # saves activation memory
GRAD_CLIP     = 1.0
ATTENTION_TYPE = "softmax"       # "softmax" (default) or "linear"
CHECKPOINT_DIR = "checkpoints"   # directory for saving checkpoints
EFFECTIVE_BATCH = MICRO_BATCH * GRAD_ACCUM  # eff. batch size (64)

DEVICE = resolve_device()
DTYPE  = resolve_dtype(DEVICE)
# bfloat16: same exponent range as fp32 — no overflow/NaN, no GradScaler needed.
# float16 caused NaN because it overflows at 65504.

print(f"Device          : {DEVICE.upper()}")
print(f"Precision       : {'BF16 + CPU-offload optimizer' if DTYPE == torch.bfloat16 else 'FP32'}")
print(f"Effective batch : {MICRO_BATCH * GRAD_ACCUM}")
print(f"Attention       : {ATTENTION_TYPE}")
print()

# ═════════════════════════════════════════════════════════════════════════════
# 4. DATA LOADER
# ═════════════════════════════════════════════════════════════════════════════

def get_batch(split="train"):
    _load_datasets()
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
# 6. OPTIMIZER / SCHEDULER / CHECKPOINT — see tinygpt.training
# ═════════════════════════════════════════════════════════════════════════════

def build_training_ctx():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    _attn_cls = LinearAttention if ATTENTION_TYPE == "linear" else None
    model = TinyGPT(vocab_size, BLOCK_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS,
                    FFN_DIM, DROPOUT, attention_cls=_attn_cls)
    n_total  = sum(p.numel() for p in model.parameters())
    n_active = n_total

    model = model.to(dtype=DTYPE, device=DEVICE)
    
    if os.environ.get("TINYGPT_COMPILE") == "1":
        print("Compiling model...")
        try:
            model = torch.compile(model, mode="default")
        except Exception as e:
            print(f"[warn] Failed to compile model: {e}")
            
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        vram_used = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU VRAM used   : {vram_used:.2f} GiB  (model weights)")

    optimizer = make_optimizer(model, lr=LR, device=DEVICE)
    gc.collect()
    if DEVICE == "cuda":
        opt_gb = n_total * 4 * 3 / 1024**3
        print(f"CPU RAM for opt : ~{opt_gb:.1f} GiB  (fp32 master + fp32 m + fp32 v)")

    print(f"Total  parameters : {n_total:>14,}")
    print(f"Active per token  : {n_active:>14,}")
    print()

    RESUME = True
    start_step = 0
    best_val   = float("inf")
    latest_ckpt = os.path.join(CHECKPOINT_DIR, "latest.pt")
    if RESUME and os.path.exists(latest_ckpt):
        try:
            print("Checkpoint found — resuming …")
            start_step, best_val = load_checkpoint(latest_ckpt, model, optimizer, attention_type=ATTENTION_TYPE)
            print()
        except ValueError as e:
            if "NaN" in str(e):
                print("[yellow]NaN checkpoint detected and removed — starting fresh.[/yellow]")
                os.remove(latest_ckpt)
                _best = os.path.join(CHECKPOINT_DIR, "best.pt")
                if os.path.exists(_best):
                    os.remove(_best)
            else:
                print(f"Checkpoint corrupted ({e}) — starting fresh")
        except Exception as e:
            print(f"Checkpoint corrupted ({e}) — starting fresh")
    else:
        if RESUME:
            print("No checkpoint found — starting fresh training")
        print()
        
    return model, optimizer, start_step, best_val

# ═════════════════════════════════════════════════════════════════════════════
# 11. TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    console.rule("[bold green]Training started")
    print()
    
    model, optimizer, start_step, best_val = build_training_ctx()

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        TextColumn("[cyan]ends {task.fields[ends_at]}"),
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
            train_loss="--.----", val_loss="--.----", lr="--.------",
            tok_s="------", ends_at="--:--",
        )

        step_start_time = time.perf_counter()
        start_wall_time = time.time()
        tokens_per_step = MICRO_BATCH * GRAD_ACCUM * BLOCK_SIZE

        for step in range(start_step + 1, MAX_ITERS + 1):

            lr = get_lr(step, LR, WARMUP_STEPS, MAX_ITERS)
            optimizer.set_lr(lr)

            optimizer.zero_grad()
            accum_loss = 0.0

            for _ in range(GRAD_ACCUM):
                x, y = get_batch("train")
                with autocast_ctx(DEVICE, DTYPE):
                    _, loss = model(x, y, use_activation_checkpoint=USE_ACTIVATION_CHECKPOINT)
                (loss / GRAD_ACCUM).backward()
                accum_loss += loss.item() / GRAD_ACCUM

            # Gradient clipping with stricter threshold to prevent explosion
            norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            if norm_before > GRAD_CLIP:
                progress.console.print(f"  [yellow]Gradient norm clipped: {norm_before:.2f} → {GRAD_CLIP}[/]", style="dim")
            optimizer.step()

            now = time.perf_counter()
            elapsed = now - step_start_time
            step_start_time = now
            tok_s = tokens_per_step / elapsed if elapsed > 0 else 0

            completed = step - start_step
            remaining_steps = total_steps - completed
            if completed > 0:
                elapsed_wall = time.time() - start_wall_time
                eta_seconds = (elapsed_wall / completed) * remaining_steps
                ends = time.strftime("%H:%M", time.localtime(time.time() + eta_seconds))
            else:
                ends = "--:--"

            progress.update(
                task, advance=1,
                train_loss=f"{accum_loss:.4f}", lr=f"{lr:.6f}",
                tok_s=f"{tok_s:,.0f}", ends_at=ends,
            )

            if step % EVAL_EVERY == 0 or step == 1:
                losses = eval_suite(model, lambda s: get_batch(s), eval_iters=EVAL_ITERS,
                                    device=DEVICE, verbose=False)
                progress.update(
                    task,
                    train_loss=f"{losses['train_loss']:.4f}",
                    val_loss=f"{losses['val_loss']:.4f}",
                    lr=f"{lr:.6f}",
                    tok_s=f"{tok_s:,.0f}",
                    ends_at=ends,
                )
                progress.console.print(
                    f"  [bold]Step {step:>5}[/]  │  "
                    f"[yellow]Train {losses['train_loss']:.4f}[/]  │  "
                    f"[cyan]Val {losses['val_loss']:.4f}  PPL {losses['val_ppl']:.1f}[/]  │  "
                    f"[magenta]LR {lr:.6f}[/]  │  "
                    f"[bold cyan]{tok_s:,.0f} tok/s[/]  │  "
                    f"[green]Acc {losses['token_accuracy']:.3f}[/]"
                )

                # ── Save checkpoints ──
                save_checkpoint(
                    step, model, optimizer,
                    losses["train_loss"], losses["val_loss"],
                    os.path.join(CHECKPOINT_DIR, "latest.pt"),
                    attention_type=ATTENTION_TYPE,
                )
                if losses["val_loss"] < best_val:
                    best_val = losses["val_loss"]
                    save_checkpoint(
                        step, model, optimizer,
                        losses["train_loss"], losses["val_loss"],
                        os.path.join(CHECKPOINT_DIR, "best.pt"),
                        attention_type=ATTENTION_TYPE,
                    )
                    progress.console.print(
                        f"  [bold green]★ New best val loss: {best_val:.4f}  (saved best.pt)[/]"
                    )

    print()
    console.rule("[bold green]Training complete")
    print()

    # ── Load best checkpoint for final evaluation ──
    best_ckpt = os.path.join(CHECKPOINT_DIR, "best.pt")
    if os.path.exists(best_ckpt):
        print("Loading best checkpoint for evaluation …")
        load_checkpoint(best_ckpt, model, optimizer, attention_type=ATTENTION_TYPE)
        print()

    # ═════════════════════════════════════════════════════════════════════════════
    # 12. TEST EVALUATION
    # ═════════════════════════════════════════════════════════════════════════════

    model.eval()
    test_losses = []
    with torch.no_grad():
        for _ in range(EVAL_ITERS):
            x, y = get_batch("test")
            _, loss = model(x, y, use_activation_checkpoint=USE_ACTIVATION_CHECKPOINT)
            test_losses.append(loss.item())
    test_loss = sum(test_losses) / len(test_losses)
    print(f"Test loss : {test_loss:.4f}")
    print()
    model.train()

    # ═════════════════════════════════════════════════════════════════════════════
    # 13. TEXT GENERATION SAMPLES
    # ═════════════════════════════════════════════════════════════════════════════

    prompts = [
        "The history of",
        "Scientists have discovered",
        "In the early twentieth century",
    ]

    print("=" * 60)
    print("Generated Text Samples")
    print("=" * 60)

    for prompt in prompts:
        output = generate(model, prompt, max_new_tokens=120, temperature=0.7)
        print(f"\nPrompt : \"{prompt}\"")
        print(f"Output : {output.strip()}")
        print()

    # ═════════════════════════════════════════════════════════════════════════════
    # 14. INTERACTIVE MODE
    # ═════════════════════════════════════════════════════════════════════════════

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
        output = generate(model, prompt, max_new_tokens=150, temperature=0.8)
        print(f"\n{output.strip()}")

    print("\nGoodbye!")
