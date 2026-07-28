"""
Tiny-GPT Chat Fine-Tuning
==========================
Fine-tunes a pre-trained TinyGPT checkpoint on DailyDialog chat data.
Uses User:/Assistant: markers to teach conversational format.

Run order:
    python prepare_chat_data.py    # once — prepares chat_data/
    python finetune.py             # fine-tune on chat data
    python chat.py                 # interactive chat
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
import tiktoken
from model import CausalSelfAttention, FeedForward, TransformerBlock, TinyGPT
from tinygpt.training import CPUOffloadAdamW, get_lr, save_checkpoint, estimate_loss
from rich.progress import (
    Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn,
    SpinnerColumn, MofNCompleteColumn,
)
from rich.console import Console

console = Console()

if __name__ == "__main__":
    # ═════════════════════════════════════════════════════════════════════════════
    # 1. LOAD CHAT DATA
    # ═════════════════════════════════════════════════════════════════════════════

    DATA_DIR = "instruction_data"
    for split in ("train", "val", "test"):
        path = os.path.join(DATA_DIR, f"{split}.bin")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"\n[ERROR] '{path}' not found.\n"
                "Run  python prepare_chat_data.py  first."
            )

    train_data = np.memmap(os.path.join(DATA_DIR, "train.bin"), dtype=np.uint16, mode="r")
    val_data   = np.memmap(os.path.join(DATA_DIR, "val.bin"),   dtype=np.uint16, mode="r")
    test_data  = np.memmap(os.path.join(DATA_DIR, "test.bin"),  dtype=np.uint16, mode="r")

    print("Instruction dataset loaded (memory-mapped)")
    print(f"  Train : {len(train_data):>12,} tokens")
    print(f"  Val   : {len(val_data):>12,} tokens")
    print(f"  Test  : {len(test_data):>12,} tokens")
    print()

    # ═════════════════════════════════════════════════════════════════════════════
    # 2. TOKENIZER
    # ═════════════════════════════════════════════════════════════════════════════

    enc        = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab

    def encode(text: str) -> list:
        return enc.encode_ordinary(text)

    def decode(ids: list) -> str:
        return enc.decode(ids)

    print(f"Tokenizer : GPT-2 BPE  (vocab {vocab_size:,})")
    print()

    # ═════════════════════════════════════════════════════════════════════════════
    # 3. HYPERPARAMETERS (tuned for fine-tuning)
    # ═════════════════════════════════════════════════════════════════════════════

    BLOCK_SIZE    = 512              # context window (tokens) — must match pre-trained
    MICRO_BATCH   = 2                # samples per GPU forward pass
    GRAD_ACCUM    = 8                # accumulate before optimizer step
    EMBED_DIM     = 768              # model width (must match pre-trained 124M)
    NUM_HEADS     = 12               # attention heads
    NUM_LAYERS    = 12               # transformer blocks
    FFN_DIM       = EMBED_DIM * 4   # 3072
    DROPOUT       = 0.1
    LR            = 2e-5             # LOW learning rate — preserve pre-trained weights
    WARMUP_STEPS  = 500
    MAX_ITERS     = 10_000           # instruction tuning steps
    EVAL_EVERY    = 1_000
    EVAL_ITERS    = 50
    USE_ACTIVATION_CHECKPOINT = True  # required for 124M on 4GB VRAM
    GRAD_CLIP     = 1.0
    CHECKPOINT_DIR = "checkpoints"
    PRETRAINED_CKPT = os.path.join(CHECKPOINT_DIR, "best.pt")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE  = torch.bfloat16 if DEVICE == "cuda" else torch.float32

    print(f"Device          : {DEVICE.upper()}")
    print(f"Precision       : {'BF16' if DTYPE == torch.bfloat16 else 'FP32'}")
    print(f"Effective batch : {MICRO_BATCH * GRAD_ACCUM}")
    print(f"Learning rate   : {LR}")
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

    # ═════════════════════════════════════════════════════════════════════════════
    # 6. OPTIMIZER / SCHEDULER / CHECKPOINT — see tinygpt.training
    # ═════════════════════════════════════════════════════════════════════════════

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ═════════════════════════════════════════════════════════════════════════════
    # 10. INSTANTIATE MODEL + LOAD PRE-TRAINED CHECKPOINT
    # ═════════════════════════════════════════════════════════════════════════════

    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    model = TinyGPT(vocab_size, BLOCK_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS,
                    FFN_DIM, DROPOUT)
    n_total = sum(p.numel() for p in model.parameters())

    model = model.to(dtype=DTYPE, device=DEVICE)
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # Force-load the pre-trained checkpoint
    if not os.path.exists(PRETRAINED_CKPT):
        raise FileNotFoundError(
            f"\n[ERROR] Pre-trained checkpoint '{PRETRAINED_CKPT}' not found.\n"
            "Run  python main.py  first to generate best.pt."
        )

    print(f"Loading pre-trained checkpoint: {PRETRAINED_CKPT}")
    ckpt = torch.load(PRETRAINED_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Loaded from step {ckpt['step']}  "
          f"(train {ckpt['train_loss']:.4f}, val {ckpt['val_loss']:.4f})")
    print()

    # Initialize optimizer
    if DEVICE == "cuda":
        optimizer = CPUOffloadAdamW(model.parameters(), lr=LR)
    else:
        _inner = torch.optim.AdamW(model.parameters(), lr=LR)
        class _Wrap:
            def __init__(self, o): self.opt = o
            def step(self):       self.opt.step()
            def zero_grad(self):  self.opt.zero_grad(set_to_none=True)
            def set_lr(self, lr):
                for pg in self.opt.param_groups: pg["lr"] = lr
            def state_dict(self):       return self.opt.state_dict()
            def load_state_dict(self, sd): self.opt.load_state_dict(sd)
        optimizer = _Wrap(_inner)

    print(f"Total  parameters : {n_total:>14,}")
    print()

    # ═════════════════════════════════════════════════════════════════════════════
    # 11. FINE-TUNING LOOP
    # ═════════════════════════════════════════════════════════════════════════════

    console.rule("[bold green]Fine-tuning started")
    print()

    start_step = 0
    best_val = float("inf")
    tokens_per_step = MICRO_BATCH * GRAD_ACCUM * BLOCK_SIZE

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
            "Fine-tuning", total=total_steps,
            train_loss="--.----", val_loss="--.----", lr="--.------",
            tok_s="------",
        )

        step_start_time = time.perf_counter()

        for step in range(start_step + 1, MAX_ITERS + 1):

            lr = get_lr(step, LR, WARMUP_STEPS, MAX_ITERS)
            optimizer.set_lr(lr)

            optimizer.zero_grad()
            accum_loss = 0.0

            for _ in range(GRAD_ACCUM):
                x, y = get_batch("train")
                with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                        enabled=(DTYPE == torch.bfloat16)):
                    _, loss = model(x, y)
                (loss / GRAD_ACCUM).backward()
                accum_loss += loss.item() / GRAD_ACCUM

            norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            if norm_before > GRAD_CLIP:
                progress.console.print(f"  [yellow]Gradient norm clipped: {norm_before:.2f} → {GRAD_CLIP}[/]", style="dim")
            optimizer.step()

            now = time.perf_counter()
            elapsed = now - step_start_time
            step_start_time = now
            tok_s = tokens_per_step / elapsed if elapsed > 0 else 0

            progress.update(
                task, advance=1,
                train_loss=f"{accum_loss:.4f}", lr=f"{lr:.6f}",
                tok_s=f"{tok_s:,.0f}",
            )

            if step % EVAL_EVERY == 0 or step == 1:
                losses = estimate_loss(model, get_batch, EVAL_ITERS)
                progress.update(
                    task,
                    train_loss=f"{losses['train']:.4f}",
                    val_loss=f"{losses['val']:.4f}",
                    lr=f"{lr:.6f}",
                    tok_s=f"{tok_s:,.0f}",
                )
                progress.console.print(
                    f"  [bold]Step {step:>5}[/]  │  "
                    f"[yellow]Train {losses['train']:.4f}[/]  │  "
                    f"[cyan]Val {losses['val']:.4f}[/]  │  "
                    f"[magenta]LR {lr:.6f}[/]  │  "
                    f"[bold cyan]{tok_s:,.0f} tok/s[/]"
                )

                # Save fine-tuned checkpoints (separate from pre-trained)
                save_checkpoint(
                    step, model, optimizer,
                    losses["train"], losses["val"],
                    os.path.join(CHECKPOINT_DIR, "finetune_latest.pt"),
                )
                if losses["val"] < best_val:
                    best_val = losses["val"]
                    save_checkpoint(
                        step, model, optimizer,
                        losses["train"], losses["val"],
                        os.path.join(CHECKPOINT_DIR, "finetune_best.pt"),
                    )
                    progress.console.print(
                        f"  [bold green]★ New best val loss: {best_val:.4f}  (saved finetune_best.pt)[/]"
                    )

    print()
    console.rule("[bold green]Fine-tuning complete")
    print()

    # ── Load best fine-tuned checkpoint for evaluation ──
    finetune_best = os.path.join(CHECKPOINT_DIR, "finetune_best.pt")
    if os.path.exists(finetune_best):
        print("Loading best fine-tuned checkpoint for evaluation ...")
        ckpt = torch.load(finetune_best, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        print(f"  Loaded from step {ckpt['step']}  "
              f"(train {ckpt['train_loss']:.4f}, val {ckpt['val_loss']:.4f})")
        print()

    # ═════════════════════════════════════════════════════════════════════════════
    # 12. TEST EVALUATION
    # ═════════════════════════════════════════════════════════════════════════════

    model.eval()
    test_losses = []
    with torch.no_grad():
        for _ in range(EVAL_ITERS):
            x, y = get_batch("test")
            _, loss = model(x, y)
            test_losses.append(loss.item())
    test_loss = sum(test_losses) / len(test_losses)
    print(f"Test loss : {test_loss:.4f}")
    print()

    print("Fine-tuning complete. Run  python chat.py  to chat with your bot!")
