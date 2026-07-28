# Tiny-GPT Refactoring Chat

## Context

Tiny-GPT is a dense GPT-2 Small (124M) model trained on FineWeb-Edu. The codebase was a monolithic research script with duplicated model definitions across `main.py`, `run.py`, `finetune.py`, and `main_deepspeed.py`.

The user requested a phased refactoring to modularize the architecture without changing behavior.

---

## Phase 1 — Extract CausalSelfAttention (BLOCKED)

**User request:** Move CausalSelfAttention into its own module. Keep every tensor shape, API, checkpoint, and output identical.

**Status:** NOT IMPLEMENTED. User stopped implementation before completion.

The plan was:
- Create `attention.py` with shared `CausalSelfAttention`
- Support two implementations via `use_flash_attention` flag
  - `True` (default): `F.scaled_dot_product_attention` with `is_causal=True` — used by training scripts
  - `False`: Manual `q @ k.T` with lazy triangular mask + `attn_drop` — used by `run.py`
- Update all four files to import from `attention.py`

**Key discovery:** Two distinct `CausalSelfAttention` variants exist:
- `main.py`, `finetune.py`, `main_deepspeed.py`: SDPA (FlashAttention), no explicit mask, no `attn_drop`
- `run.py`: Manual attention with explicit triangular mask, has `attn_drop` layer

---

## Phase 2 — Abstract Attention Interface (BLOCKED)

**User request:** Create an abstract Attention interface. Existing CausalSelfAttention must implement it.

**Status:** DESIGN ONLY. Never implemented. User jumped to Phase 3.

**Design:**
```
attention_interface.py        (NEW — abstract base class)
├── class Attention(ABC)
│   └── @abstractmethod forward(self, x) -> Tensor

main.py / run.py / finetune.py / main_deepspeed.py
├── from attention_interface import Attention
├── class CausalSelfAttention(Attention): ...
```

---

## Phase 3 — LinearAttention Implementation (COMPLETED)

**User request:** Implement LinearAttention as a second implementation of the Attention interface. Add `ATTENTION_TYPE` config (softmax/linear, default softmax). Benchmark speed, VRAM, loss.

**Status:** COMPLETED.

### Files modified/created

| File | Change |
|------|--------|
| `main.py` | Added `ATTENTION_TYPE = "softmax"` config, import `LinearAttention`, TransformerBlock conditional, checkpoint save/load handles attention type mismatch |
| `linear_attention.py` | **NEW.** `LinearAttention` class with elu+1 kernel, O(N) causal state accumulation |
| `benchmark_attention.py` | **NEW.** Benchmark script comparing both attention types |

### Configuration

```python
ATTENTION_TYPE = "softmax"  # or "linear"  (in main.py hyperparameters)
```

### Checkpoint compatibility

- `save_checkpoint` stores `attention_type` in checkpoint dict
- `load_checkpoint` detects type mismatch, skips attention weights (`strict=False`), prints warning
- Old checkpoints (no `attention_type` field) default to `"softmax"`

### Benchmark results (5 steps, batch 2, CUDA bfloat16)

```
              softmax         linear          ratio
Params:       124,009,728     124,009,728     identical
Speed:        177.9 ms/step   3932.1 ms/step  22.1x slower
Throughput:   5,756 tok/s     260 tok/s
Peak VRAM:    1.62 GiB        2.89 GiB        1.78x more
Avg loss:     260.33          258.89          similar
```

### LinearAttention architecture

```python
class LinearAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        # Separate Q, K, V projections (not fused)
        self.q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        # elu+1 kernel feature map
        # Causal state accumulation: O(N) per head
        # kv += k_t * v_t^T, state_k += k_t
        # out_t = (kv @ q_t) / (state_k . q_t + eps)
```

---

## Refactoring Rules (from user)

1. Never rewrite the repository
2. Never touch more than ONE subsystem in a single phase
3. Every phase must compile
4. Every phase must pass all existing tests
5. Every phase must be reversible with a single git revert
6. Never optimize and refactor in the same phase
7. Never implement future features until the architecture is ready
8. Behavior must remain identical

---

## Pending Phases

- Phase 1 (Extract attention) — needs implementation
- Phase 2 (Abstract interface) — needs implementation
- Future phases: FlashAttention changes, HybridAttention, LinearAttention optimization — blocked until architecture is ready
