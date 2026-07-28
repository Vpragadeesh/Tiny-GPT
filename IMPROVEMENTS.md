# Improvements Applied

## Phase 1: Critical Fixes
- [x] Fixed `chat.py` crash — `TinyGPT()` zero-argument constructor replaced with full parameterized call
- [x] Added `__main__` guard to `chat.py` — prevents chat loop from running on import
- [x] Added `__main__` guard to `finetune.py` — prevents training from running on import
- [x] Fixed checkpoint weight tying — `head.weight` keys are stripped on load to respect tying
- [x] Unified checkpoint keys — all scripts now use `"model_state"` key (was `"model"` in `main.py` vs `"model_state"` in `main_deepspeed.py`)
- [x] Removed duplicate `linear_attention.py` — identical copy of `tinygpt/attention/linear.py`, deleted

## Phase 2: Training Optimization
- [x] **Increased effective batch size**: MICRO_BATCH 2→4, GRAD_ACCUM 8→16 (effective batch 16→64)
  - Larger batches = more stable gradients, better convergence
- [x] **Gradient accumulation**: Already implemented. Verified correct pattern (`loss / GRAD_ACCUM` per micro-batch)
- [x] **LR schedule**: Already implemented (linear warmup + cosine decay via `get_lr()`). Verified correct
- [x] **Eval suite**: Created `eval_suite.py` tracking loss, perplexity, token accuracy, generation diversity
- [x] **Integrated eval_suite**: Training progress now shows PPL and accuracy alongside loss
- [x] **Data quality checks**: `prepare_data.py` now filters low-diversity sequences (<50 unique tokens), logs stats

## Phase 3: Quantization Preparation
(Skipped — deferred)

## Phase 4: Validation
- [x] Created `tests/` directory with unit tests for model forward, checkpoint save/load, chat init, eval_suite import

## Phase 5: Documentation
- [x] Updated `README.md` — corrected from stale MoE 0.5B description to accurate 124M dense GPT-2
- [x] Created `IMPROVEMENTS.md` — this file

## Results
| Metric | Before | After |
|--------|--------|-------|
| chat.py import | CRASH | OK |
| finetune.py import | Triggers training | No side effects |
| Checkpoint key | `"model"` / `"model_state"` mismatch | Unified `"model_state"` |
| Effective batch | 16 | 64 |
| Eval metrics | Loss only | Loss + PPL + Accuracy + Diversity |
| Data quality filtering | None | Low-diversity skip + stats |
| Unit tests | None | 4 tests |
| README accuracy | MoE 0.5B (wrong) | Dense 124M (correct) |
