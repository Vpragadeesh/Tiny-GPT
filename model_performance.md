# Tiny-GPT Model Performance Report

## Architecture

| Parameter | Value |
|-----------|-------|
| Architecture | Dense GPT-2 Small |
| Total parameters | 124,009,728 |
| Trainable parameters | 124,009,728 |
| Context window | 512 tokens |
| Vocabulary size | 50,257 (GPT-2 BPE) |
| Embedding dim | 768 |
| Attention heads | 12 (head_dim = 64) |
| Transformer layers | 12 |
| FFN dim | 3,072 (4x embed_dim) |
| Dropout | 0.1 |
| Weight tying | head ↔ tok_emb |

## Parameter Breakdown

| Component | Parameters | % of Total |
|-----------|-----------|------------|
| Embeddings (tok + pos) | 38,990,592 | 31.4% |
| Attention (12 blocks) | 28,311,552 | 22.8% |
| FFN (12 blocks) | 56,669,184 | 45.7% |
| LayerNorm (24 + 1) | 38,400 | 0.0% |

## Model Size

| Format | Size |
|--------|------|
| FP32 | 473.1 MB |
| BF16 (training) | 236.5 MB |
| Checkpoint (with optimizer) | ~1.7 GB |

## Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Peak LR | 1.5e-4 |
| Warmup steps | 500 |
| Max iterations | 500,000 |
| Micro batch size | 2 |
| Gradient accumulation | 8 |
| Effective batch | 16 |
| Optimizer | CPUOffloadAdamW (fp32 master on CPU) |
| LR schedule | Linear warmup → cosine decay to 10% |
| Gradient clipping | 1.0 |
| Precision | BF16 (CUDA) / FP32 (CPU) |
| Activation checkpointing | Enabled |

## Checkpoint Status

| Checkpoint | Step | Train Loss | Val Loss | Status |
|------------|------|------------|----------|--------|
| best.pt | 1 | 10.9938 | 10.9850 | Untrained (initial weights) |
| latest.pt | 1 | 10.9938 | 10.9850 | Untrained (initial weights) |

> **Note:** Loss ≈ 10.99 ≈ ln(50,257) ≈ random baseline. Model has not been trained.

## Benchmark: Softmax vs Linear Attention

| Metric | Softmax (SDPA) | Linear | Ratio |
|--------|---------------|--------|-------|
| Parameters | 124,009,728 | 124,009,728 | 1.0x |
| Speed | 177.9 ms/step | 3,932.1 ms/step | 22.1x slower |
| Throughput | 5,756 tok/s | 260 tok/s | — |
| Peak VRAM | 1.62 GiB | 2.89 GiB | 1.78x more |
| Avg loss (random data) | 260.33 | 258.89 | Similar |

## Attention Implementations

| Variant | Where Used | Mechanism |
|---------|-----------|-----------|
| SDPA (default) | main.py, finetune.py, main_deepspeed.py | `F.scaled_dot_product_attention` with `is_causal=True` |
| Manual | run.py | Explicit `q @ k.T` with triangular mask buffer |
| Linear | (optional via `ATTENTION_TYPE="linear"`) | O(N) kernel attention with elu+1 feature map |

## Known Issues

1. **Training not started:** Checkpoints contain only initial random weights (step 1). Run `python main.py` to train.
2. **Linear attention is slow:** The O(N) loop over 512 tokens is sequential — 22x slower than SDPA. Needs vectorized implementation for production use.
3. **No validation:** Model has not been evaluated on any downstream task.

## Code Structure (Post-Refactoring)

```
tinygpt/
├── attention/
│   ├── causal.py        # CausalSelfAttention (SDPA + manual)
│   └── linear.py        # LinearAttention (elu+1 kernel)
├── layers/
│   ├── feedforward.py   # FeedForward (GELU FFN)
│   └── transformer_block.py  # TransformerBlock (pre-norm)
├── training/
│   ├── optimizer.py     # CPUOffloadAdamW
│   ├── scheduler.py     # Linear warmup → cosine decay
│   ├── checkpoint.py    # Save/load with attention type handling
│   └── evaluation.py    # Loss estimation
└── generation/
    └── (pending Subsystem 4)

model.py                 # TinyGPT (imports from tinygpt.*)
main.py                  # Training entry point
run.py                   # Inference entry point
finetune.py              # Fine-tuning entry point
main_deepspeed.py        # DeepSpeed training entry point
```
