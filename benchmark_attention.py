"""
Benchmark: softmax vs linear attention.
Compares speed, VRAM, and loss for a short training run.

Usage:
    python benchmark_attention.py          # runs both, compares
    python benchmark_attention.py --steps 20
"""

import os
import sys
import time
import argparse
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Shared config (must match main.py) ──

BLOCK_SIZE  = 512
EMBED_DIM   = 768
NUM_HEADS   = 12
NUM_LAYERS  = 12
FFN_DIM     = EMBED_DIM * 4
DROPOUT     = 0.1
LR          = 1.5e-4
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE       = torch.bfloat16 if DEVICE == "cuda" else torch.float32
VOCAB_SIZE  = 50257  # GPT-2


def measure_vram():
    if DEVICE != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**3


def build_model(attention_type):
    """Build TinyGPT with the specified attention type."""
    from tinygpt.attention.linear import LinearAttention

    class CausalSelfAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_heads  = NUM_HEADS
            self.head_dim = EMBED_DIM // NUM_HEADS
            self.qkv      = nn.Linear(EMBED_DIM, 3 * EMBED_DIM, bias=False)
            self.proj      = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
            self.proj_drop = nn.Dropout(DROPOUT)
        def forward(self, x):
            B, T, C = x.shape
            qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            out = out.transpose(1, 2).reshape(B, T, C)
            return self.proj_drop(self.proj(out))

    class FeedForward(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1  = nn.Linear(EMBED_DIM, FFN_DIM)
            self.w2  = nn.Linear(FFN_DIM, EMBED_DIM)
            self.act = nn.GELU()
            self.drop = nn.Dropout(DROPOUT)
        def forward(self, x):
            return self.drop(self.w2(self.act(self.w1(x))))

    class TransformerBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(EMBED_DIM)
            if attention_type == "linear":
                self.attn = LinearAttention(EMBED_DIM, NUM_HEADS, DROPOUT)
            else:
                self.attn = CausalSelfAttention()
            self.ln2 = nn.LayerNorm(EMBED_DIM)
            self.ffn = FeedForward()
        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
            return x

    class TinyGPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
            self.pos_emb = nn.Embedding(BLOCK_SIZE, EMBED_DIM)
            self.drop    = nn.Dropout(DROPOUT)
            self.blocks  = nn.ModuleList([TransformerBlock() for _ in range(NUM_LAYERS)])
            self.ln_f    = nn.LayerNorm(EMBED_DIM)
            self.head    = nn.Linear(EMBED_DIM, VOCAB_SIZE, bias=False)
            self.head.weight = self.tok_emb.weight
        def forward(self, idx, targets=None):
            B, T = idx.shape
            x = self.drop(self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device)))
            for block in self.blocks:
                x = block(x)
            logits = self.head(self.ln_f(x))
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
            return logits, loss

    return TinyGPT()


def benchmark(attention_type, steps=10, batch_size=2):
    """Run a short training loop and measure metrics."""
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    model = build_model(attention_type)
    model = model.to(dtype=DTYPE, device=DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # Warmup
    x = torch.randint(0, VOCAB_SIZE, (batch_size, BLOCK_SIZE), device=DEVICE)
    y = torch.randint(0, VOCAB_SIZE, (batch_size, BLOCK_SIZE), device=DEVICE)
    with torch.amp.autocast("cuda", dtype=DTYPE, enabled=(DTYPE == torch.bfloat16)):
        _, loss = model(x, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Timed steps
    losses = []
    times = []
    for _ in range(steps):
        x = torch.randint(0, VOCAB_SIZE, (batch_size, BLOCK_SIZE), device=DEVICE)
        y = torch.randint(0, VOCAB_SIZE, (batch_size, BLOCK_SIZE), device=DEVICE)
        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=DTYPE, enabled=(DTYPE == torch.bfloat16)):
            _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        losses.append(loss.item())
        times.append(t1 - t0)

    vram = measure_vram()
    avg_time = np.mean(times)
    avg_loss = np.mean(losses)
    tokens_per_sec = (batch_size * BLOCK_SIZE) / avg_time

    return {
        "params": n_params,
        "avg_time_ms": avg_time * 1000,
        "tokens_per_sec": tokens_per_sec,
        "vram_gb": vram,
        "avg_loss": avg_loss,
        "final_loss": losses[-1],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10, help="Training steps per attention type")
    parser.add_argument("--batch", type=int, default=2, help="Micro batch size")
    args = parser.parse_args()

    print(f"Device: {DEVICE.upper()}  |  DTYPE: {DTYPE}")
    print(f"Steps: {args.steps}  |  Batch: {args.batch}")
    print()

    results = {}
    for attn_type in ("softmax", "linear"):
        print(f"── Benchmarking {attn_type} attention ──")
        r = benchmark(attn_type, steps=args.steps, batch_size=args.batch)
        results[attn_type] = r
        print(f"  Params    : {r['params']:>14,}")
        print(f"  Avg time  : {r['avg_time_ms']:>10.1f} ms/step")
        print(f"  Throughput: {r['tokens_per_sec']:>10,.0f} tok/s")
        if r['vram_gb'] > 0:
            print(f"  Peak VRAM : {r['vram_gb']:>10.2f} GiB")
        print(f"  Avg loss  : {r['avg_loss']:>10.4f}")
        print(f"  Final loss: {r['final_loss']:>10.4f}")
        print()

    # ── Comparison ──
    s, l = results["softmax"], results["linear"]
    print("═══ Comparison ═══")
    print(f"  Params    : softmax {s['params']:,}  |  linear {l['params']:,}")
    print(f"  Avg time  : softmax {s['avg_time_ms']:.1f} ms  |  linear {l['avg_time_ms']:.1f} ms  "
          f"({l['avg_time_ms']/s['avg_time_ms']:.2f}x)")
    print(f"  Throughput: softmax {s['tokens_per_sec']:,.0f} tok/s  |  linear {l['tokens_per_sec']:,.0f} tok/s")
    if s['vram_gb'] > 0:
        print(f"  Peak VRAM : softmax {s['vram_gb']:.2f} GiB  |  linear {l['vram_gb']:.2f} GiB  "
              f"({l['vram_gb']/s['vram_gb']:.2f}x)")
    print(f"  Avg loss  : softmax {s['avg_loss']:.4f}  |  linear {l['avg_loss']:.4f}")


if __name__ == "__main__":
    main()
