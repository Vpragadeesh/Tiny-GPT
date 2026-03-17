"""
MoE GPT – 0.5 Billion Parameter Language Model
================================================
Mixture-of-Experts GPT trained on WikiText-2.
Fits on GPUs with as little as 4 GB VRAM via:
  - FP16 model weights on GPU  (~1 GB)
  - CPU-offloaded AdamW         (optimizer states on RAM, not VRAM)
  - Gradient checkpointing      (recompute activations to save memory)

Architecture
  12 Transformer layers  ×  (12-head attention  +  MoE FFN)
  8 expert FFNs per layer, top-2 routing
  Total params  ≈ 521 M   |   Active per token  ≈ 180 M

Run order:
    pip install torch tiktoken numpy datasets
    python prepare_data.py          # once — downloads WikiText-2
    python main.py                  # train + generate
"""

import os
import math
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
import tiktoken
from rich.progress import (
    Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn,
    SpinnerColumn, MofNCompleteColumn,
)
from rich.console import Console
from rich.table import Table
from rich import print as rprint

console = Console()

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

BLOCK_SIZE    = 128              # context window (tokens)
MICRO_BATCH   = 2                # samples per GPU forward pass (tiny for VRAM)
GRAD_ACCUM    = 8                # accumulate before optimizer step → eff. batch 16
EMBED_DIM     = 768              # model width
NUM_HEADS     = 12               # attention heads
NUM_LAYERS    = 12               # transformer blocks
NUM_EXPERTS   = 8                # expert FFNs per MoE layer
TOP_K         = 2                # experts activated per token
FFN_DIM       = EMBED_DIM * 4   # 3 072  (expert hidden dim)
DROPOUT       = 0.1
LR            = 1.5e-4           # peak learning rate (reduced from 3e-4 to prevent NaN)
WARMUP_STEPS  = 500              # increased warmup for stability
MAX_ITERS     = 10000            # extended training to 10k steps
EVAL_EVERY    = 500
EVAL_ITERS    = 50
AUX_LOSS_W    = 0.01             # load-balancing auxiliary loss weight
GRAD_CLIP     = 1.0
CHECKPOINT_DIR = "checkpoints"   # directory for saving checkpoints

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if DEVICE == "cuda" else torch.float32
# bfloat16: same exponent range as fp32 — no overflow/NaN, no GradScaler needed.
# float16 caused NaN because it overflows at 65504.

print(f"Device          : {DEVICE.upper()}")
print(f"Precision       : {'BF16 + CPU-offload optimizer' if DTYPE == torch.bfloat16 else 'FP32'}")
print(f"Effective batch : {MICRO_BATCH * GRAD_ACCUM}")
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
# 6. CPU-OFFLOAD OPTIMIZER
#    Hand-rolled AdamW with fp32 master weights but fp16 momentum/variance.
#    Saves ~2 GB CPU RAM compared to using torch.optim.AdamW (all fp32).
#    GPU VRAM cost ≈ 1 GB  (only fp16 model weights + grads).
#
#    Memory breakdown for 520 M params:
#      fp32 master weights : ~2.0 GB
#      fp16 momentum       : ~1.0 GB
#      fp16 variance       : ~1.0 GB
#      Total CPU RAM       : ~4.1 GB  (was ~6.2 GB with all-fp32 AdamW)
# ═════════════════════════════════════════════════════════════════════════════

class CPUOffloadAdamW:
    """
    AdamW with ALL state (master weights, momentum, variance) in fp32 on CPU.
    fp16 m/v was the culprit for NaN — Adam variance accumulates squared
    gradients that easily exceed fp16 max (65504) → overflow → NaN.
    GPU holds only fp16 model weights + fp16 gradients (~1 GB VRAM).
    CPU RAM: fp32 master(2 GB) + fp32 m(2 GB) + fp32 v(2 GB) ≈ 6.2 GB.
    Expose param_groups so torch.amp.GradScaler.unscale_() works correctly.
    """

    def __init__(self, gpu_params, lr=3e-4, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.01):
        self.gpu_params = list(gpu_params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.t = 0

        # fp32 master copies + fp32 momentum/variance on CPU
        self.master = [p.data.float().cpu() for p in self.gpu_params]
        self.m = [torch.zeros_like(mp) for mp in self.master]   # fp32
        self.v = [torch.zeros_like(mp) for mp in self.master]   # fp32

        # GradScaler compatibility: unscale_() iterates param_groups
        self.param_groups = [{"params": self.gpu_params}]

    def step(self):
        self.t += 1
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t

        for i, gp in enumerate(self.gpu_params):
            if gp.grad is None:
                continue
            g = gp.grad.data.float().cpu()   # fp16 grad → fp32

            # Decoupled weight decay
            self.master[i].mul_(1.0 - self.lr * self.wd)

            # Adam moments (all fp32 — no overflow risk)
            self.m[i].mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
            self.v[i].mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)

            # Bias-corrected parameter update
            self.master[i].addcdiv_(
                self.m[i] / bc1,
                (self.v[i] / bc2).sqrt_().add_(self.eps),
                value=-self.lr,
            )

            # Push updated fp32 weights → GPU fp16
            gp.data.copy_(self.master[i])

    def zero_grad(self):
        for gp in self.gpu_params:
            gp.grad = None

    def set_lr(self, lr):
        self.lr = lr

    def state_dict(self):
        return {"t": self.t, "master": self.master, "m": self.m, "v": self.v}

    def load_state_dict(self, sd):
        self.t = sd["t"]
        self.master = sd["master"]
        self.m = sd["m"]
        self.v = sd["v"]
        for gp, mp in zip(self.gpu_params, self.master):
            gp.data.copy_(mp.data)

# ═════════════════════════════════════════════════════════════════════════════
# 7. CHECKPOINT HELPERS
# ═════════════════════════════════════════════════════════════════════════════

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save_checkpoint(step, model, optimizer, train_loss, val_loss, path):
    """Save model + optimizer + training state to disk."""
    torch.save({
        "step":        step,
        "model":       model.state_dict(),
        "optimizer":   optimizer.state_dict(),
        "train_loss":  train_loss,
        "val_loss":    val_loss,
    }, path)

def load_checkpoint(path, model, optimizer):
    """Load checkpoint and return the step to resume from."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"  Resumed from step {ckpt['step']}  "
          f"(train {ckpt['train_loss']:.4f}, val {ckpt['val_loss']:.4f})")
    return ckpt["step"], ckpt["val_loss"]

# ═════════════════════════════════════════════════════════════════════════════
# 8. LEARNING-RATE SCHEDULE  (linear warmup → cosine decay to 10 %)
# ═════════════════════════════════════════════════════════════════════════════

def get_lr(step):
    if step < WARMUP_STEPS:
        return LR * step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, MAX_ITERS - WARMUP_STEPS)
    return LR * 0.1 + 0.5 * LR * 0.9 * (1 + math.cos(math.pi * progress))

# ═════════════════════════════════════════════════════════════════════════════
# 9. LOSS ESTIMATION
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(EVAL_ITERS):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out

# ═════════════════════════════════════════════════════════════════════════════
# 10. INSTANTIATE MODEL + OPTIMIZER
# ═════════════════════════════════════════════════════════════════════════════

if DEVICE == "cuda":
    torch.cuda.empty_cache()

# ── Delete any NaN-poisoned checkpoints before loading ──
_nan_guard = os.path.join(CHECKPOINT_DIR, "latest.pt")
if os.path.exists(_nan_guard):
    try:
        _c = torch.load(_nan_guard, map_location="cpu", weights_only=False)
        if _c.get("val_loss") != _c.get("val_loss"):  # nan != nan
            os.remove(_nan_guard)
            _best = os.path.join(CHECKPOINT_DIR, "best.pt")
            if os.path.exists(_best):
                os.remove(_best)
            print("[yellow]NaN checkpoint detected and removed — starting fresh.[/yellow]")
    except Exception:
        pass

model = MoEGPT()
n_total  = sum(p.numel() for p in model.parameters())
_expert1 = sum(p.numel() for p in model.blocks[0].moe.experts[0].parameters())
n_active = n_total - _expert1 * (NUM_EXPERTS - TOP_K) * NUM_LAYERS

# Move to GPU in fp16  (or stay fp32 on CPU)
model = model.to(dtype=DTYPE, device=DEVICE)
gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()
    vram_used = torch.cuda.memory_allocated() / 1024**3
    print(f"GPU VRAM used   : {vram_used:.2f} GiB  (model weights)")

if DEVICE == "cuda":
    # Initialize optimizer AFTER config changes so it uses the new LR
    optimizer = CPUOffloadAdamW(model.parameters(), lr=LR)
    gc.collect()
    opt_gb = n_total * 4 * 3 / 1024**3   # fp32 master + fp32 m + fp32 v
    print(f"CPU RAM for opt : ~{opt_gb:.1f} GiB  (fp32 master + fp32 m + fp32 v)")
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
print(f"Active per token  : {n_active:>14,}")
print()

# ── Auto-resume from latest checkpoint ──
RESUME = True  # Automatically resume from latest.pt if it exists
start_step = 0
best_val   = float("inf")
latest_ckpt = os.path.join(CHECKPOINT_DIR, "latest.pt")
if RESUME and os.path.exists(latest_ckpt):
    try:
        _c = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
        # Skip NaN-poisoned checkpoints
        if _c.get("val_loss") != _c.get("val_loss") or _c.get("train_loss") != _c.get("train_loss"):
            print("Checkpoint has NaN losses — deleting and starting fresh")
            os.remove(latest_ckpt)
        else:
            print("Checkpoint found — resuming …")
            start_step, best_val = load_checkpoint(latest_ckpt, model, optimizer)
            print()
    except Exception as e:
        print(f"Checkpoint corrupted ({e}) — starting fresh")
else:
    if RESUME:
        print("No checkpoint found — starting fresh training")
    print()

# ═════════════════════════════════════════════════════════════════════════════
# 11. TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════════════

console.rule("[bold green]Training started")
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
    TextColumn("[yellow]loss {task.fields[train_loss]}"),
    TextColumn("[cyan]val {task.fields[val_loss]}"),
    TextColumn("[magenta]lr {task.fields[lr]}"),
    console=console,
    refresh_per_second=4,
) as progress:
    total_steps = MAX_ITERS - start_step
    task = progress.add_task(
        "Training", total=total_steps,
        train_loss="--.----", val_loss="--.----", lr="--.------",
    )

    for step in range(start_step + 1, MAX_ITERS + 1):

        lr = get_lr(step)
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

        # Gradient clipping with stricter threshold to prevent explosion
        norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        if norm_before > GRAD_CLIP:
            progress.console.print(f"  [yellow]Gradient norm clipped: {norm_before:.2f} → {GRAD_CLIP}[/]", style="dim")
        optimizer.step()

        progress.update(
            task, advance=1,
            train_loss=f"{accum_loss:.4f}", lr=f"{lr:.6f}",
        )

        if step % EVAL_EVERY == 0 or step == 1:
            losses = estimate_loss()
            progress.update(
                task,
                train_loss=f"{losses['train']:.4f}",
                val_loss=f"{losses['val']:.4f}",
                lr=f"{lr:.6f}",
            )
            progress.console.print(
                f"  [bold]Step {step:>5}[/]  │  "
                f"[yellow]Train {losses['train']:.4f}[/]  │  "
                f"[cyan]Val {losses['val']:.4f}[/]  │  "
                f"[magenta]LR {lr:.6f}[/]"
            )

            # ── Save checkpoints ──
            save_checkpoint(
                step, model, optimizer,
                losses["train"], losses["val"],
                os.path.join(CHECKPOINT_DIR, "latest.pt"),
            )
            if losses["val"] < best_val:
                best_val = losses["val"]
                save_checkpoint(
                    step, model, optimizer,
                    losses["train"], losses["val"],
                    os.path.join(CHECKPOINT_DIR, "best.pt"),
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
    load_checkpoint(best_ckpt, model, optimizer)
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
    output = model.generate(prompt, max_new_tokens=120, temperature=0.7)
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
    output = model.generate(prompt, max_new_tokens=150, temperature=0.8)
    print(f"\n{output.strip()}")

print("\nGoodbye!")
