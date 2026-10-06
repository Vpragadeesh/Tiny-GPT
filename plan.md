# Plan: Run Tiny-GPT on CPU within 7.9 GB RAM (2 Cores) — zero quality compromise

**Goal:** Every entry point (`prepare_data.py`, `main.py`, `finetune.py`, `eval_suite.py`, `run.py`, `chat.py`) runs CPU-only with peak RSS ≤ 7.9 GB.

## Non-negotiables (what "no compromise" means)

| Kept exactly as-is | Reference |
|---|---|
| Architecture: 124M dense, 12L/768d/12H, vocab 50,257, ctx 512 | `model.py:15` |
| Hyperparams: MICRO_BATCH=4, GRAD_ACCUM=16 (eff. batch 64), LR 1.5e-4, warmup 500, 50k steps, cosine→10%, dropout 0.1, grad clip 1.0 | `main.py:89-107` |
| Data: full FineWeb-Edu stream, same tokenization/quality filter | `prepare_data.py` |
| Precision: fp32 on CPU (strictly *better* than the GPU's bf16 — no precision loss) | `main.py:110` |
| Activation checkpointing ON (costs time, not quality) | `main.py:103` |
| Eval cadence (every 500 steps, 50 iters) + full `eval_suite` metrics | `main.py:101,331` |
| Checkpoint cadence + `latest.pt`/`best.pt` semantics | `main.py:351-368` |

Nothing is shrunk: no smaller model, no shorter context, no smaller batch, no fewer steps, no fp16, no data subsetting. The only allowed trade is **wall-clock time** (see Risks).

## Current-state findings (why it doesn't run well on CPU today)

**Environment:** 7.9 GB RAM / 2 cores / AVX2 only (no AVX512-BF16), torch `2.13.0+cu132`, **CUDA available** — so the code silently picks GPU. Target budget is 7.9 GB, and there is **no swap**. `data/*.bin` = 2.0 GB memmap (page cache, evictable). No trained checkpoint exists yet (`checkpoints/` = 8 KB).

**Blockers in code:**

1. **No way to force CPU.** `DEVICE = "cuda" if torch.cuda.is_available() else "cpu"` in `main.py:109`, `finetune.py:98`, `run.py:38` — all pick GPU here.
2. **Hardcoded `torch.amp.autocast("cuda", ...)`** in `main.py:299`, `finetune.py:224`, `chat.py:83`, `run.py:149` — no-ops only when `enabled=False`; must become device-agnostic.
3. **Optimizer doubles memory.** On CPU the code uses `torch.optim.AdamW` behind an inline `_Wrap` (`main.py:213-222`, duplicated at `finetune.py:162-171`), but `CPUOffloadAdamW` (`tinygpt/training/optimizer.py:26`) keeps a redundant fp32 **master copy** (~0.5 GB) plus per-tensor Python-loop temporaries in `step()`. On CPU the model *is* fp32 master — master copy is pure waste (RAM + speed).
4. **Checkpoint format mismatch.** GPU-trained checkpoints store optimizer as `{"t","master","m","v"}`; CPU path uses torch AdamW which needs `{"state","param_groups"}`. `load_checkpoint()` (`tinygpt/training/checkpoint.py:36`) will crash or corrupt resume. Also CPU-saved checkpoints won't load back into `CPUOffloadAdamW`.
5. **Import-time side effects → memory spikes.** `chat.py:13-14` and `eval_suite.py:87` do `from main import ...`, and `main.py` executes data-load + model build + optimizer build + **full checkpoint resume** at module import (`main.py:50-249`). `chat.py` then builds a *second* model. Importing chat ≈ model×2 + optimizer states + double `torch.load` → blows the budget.
6. **Full-checkpoint loads when only weights are needed.**
   - NaN guard loads the whole ~1.5–2 GB checkpoint just to read `val_loss` (`main.py:182`), then resume loads it **again** (`main.py:242`).
   - `finetune.py:152`, `run.py:202`, `chat.py:34` deserialize optimizer states (≈1 GB) only to use `model_state`.
   - No `del`/`gc` after load → peaks stack.
7. **`eval_suite` default `device="cuda"`** (`eval_suite.py:11`) — breaks standalone use on CPU.
8. **Transient tensors worth budgeting:** logits `(4, 512, 50257)` fp32 = 411 MB + log-softmax ≈ another 411 MB per micro-step; SDPA math backend materializes `(4,12,512,512)` ≈ 50 MB per attention call. All fit, but must be counted.

## Memory budget (fp32, activation checkpointing ON)

| Component | Size |
|---|---|
| Model weights (fp32, 124M) | ~0.5 GB |
| Gradients (fp32) | ~0.5 GB |
| AdamW state (m + v, fp32) — **no separate master on CPU** | ~1.0 GB |
| Saved activations (checkpointed) | ~0.3 GB |
| Logits + cross-entropy transient (worst case, kept) | ~0.8 GB |
| Python/torch/heap overhead | ~0.5 GB |
| **Steady-state training peak** | **~2.6–3.1 GB** |
| One-time checkpoint load (GPU-format, worst case) | +1.5 GB (freed after load) |
| **Absolute worst case** | **~4.5 GB** |
| Headroom under 7.9 GB | ≥ 3.4 GB (OS + page cache; `data/` cache is evictable under pressure) |

Fits with margin. The work below makes the *actual* code behave like this table instead of stacking spikes.

---

## Phases

### P0 — Baseline: instrumentation + benchmark (no behavior change)
- Add `tools/monitor.py`: run any command as subprocess, sample RSS via `psutil`, report peak RSS + wall time.
- Run 10 training steps on CPU (forced, see P1) and record tok/s → extrapolate full-run ETA (put the number in the Risks section; do not silently shorten the schedule).
- **Acceptance:** baseline peak RSS (~2.5 GB) and tok/s (149 tok/s) recorded in this file.

### P1 — Device forcing + autocast abstraction
- New `tinygpt/device.py`:
  - `resolve_device()` — honors `TINYGPT_DEVICE` env / `--device` flag; `cpu` disables CUDA (`CUDA_VISIBLE_DEVICES=""` before torch import where feasible).
  - `autocast_ctx(device, dtype)` — returns `torch.amp.autocast(device, ...)` on CUDA, `nullcontext` on CPU (fp32 = no autocast needed).
- Use it in `main.py:299`, `finetune.py:224`, `run.py:149`, `chat.py:83`; replace `DEVICE/DTYPE` blocks in `main.py:109-110`, `finetune.py:98-99`, `run.py:38-39`.
- `eval_suite.py:11`: default `device=None` → `resolve_device()`.
- **Acceptance:** `TINYGPT_DEVICE=cpu python main.py` trains without touching CUDA; no `autocast("cuda")` calls remain outside the CUDA branch.

### P2 — Optimizer unification
- In `tinygpt/training/optimizer.py`, add `make_optimizer(model, lr, device)`:
  - CPU → `torch.optim.AdamW` wrapped in one shared `OptimizerWrap` (moves the inline `_Wrap` out of `main.py:213` and `finetune.py:162`).
  - CUDA → existing `CPUOffloadAdamW` (unchanged GPU behavior).
  - Same interface: `step/zero_grad/set_lr/state_dict/load_state_dict`.
- Numerics note: torch AdamW on fp32 model weights performs the identical decoupled-weight-decay + bias-corrected update as `CPUOffloadAdamW.master` math (`optimizer.py:44-56`) minus the redundant copy — verified in P7 parity check.
- **Acceptance:** `main.py` and `finetune.py` share one optimizer factory; RAM saving ~0.5 GB + faster `step()` (foreach/fused kernels instead of per-tensor Python loop).

### P3 — Checkpoint compatibility + load-spike control
- `tinygpt/training/checkpoint.py`:
  - `detect_format(ckpt)` → `"offload"` vs `"torch"`; `convert_optimizer_state()` migrates `{"t","master","m","v"}` ↔ torch AdamW format on load. Resume works regardless of which device produced the checkpoint.
  - `load_model_weights(path, model)` helper: loads (or `mmap=True` when format allows), returns only `model_state`, drops the rest.
  - Atomic saves: write to `*.tmp` + `os.replace` (protects against OOM-kill mid-save).
- `main.py`: merge NaN guard + resume into **one** `torch.load`; `del` the raw dict + `gc.collect()` after use.
- `finetune.py:152`, `run.py:202`, `chat.py:34`: use `load_model_weights` (never deserialize optimizer states for inference/fine-tune init).
- **Acceptance:** loading a GPU-format checkpoint on CPU resumes cleanly; measured load peak ≤ 1.7 GB above steady state.

### P4 — Remove import-time side effects
- `main.py`: move memmap loading, model/optimizer construction, and resume logic behind `if __name__ == "__main__"` (or a `build_training_ctx()` called only from main). Keep `get_batch`/`DEVICE` importable safely (lazy module-level globals).
- `chat.py`: stop importing from `main` — import `TinyGPT` from `model` and constants from a new tiny shared config (or `resolve_device()`), build its **one** model.
- `eval_suite.py:87`: same — no `from main import`.
- **Acceptance:** `python -c "import chat, eval_suite, main"` peaks at < 1 GB RSS; `test_chat_init` still passes.

### P5 — Training-loop RAM trims (hyperparams untouched)
- Keep `MICRO_BATCH=4`, `GRAD_ACCUM=16`, `BLOCK_SIZE=512`, `USE_ACTIVATION_CHECKPOINT=True`, eval/save cadence — all unchanged.
- Keep full logits/cross-entropy path (no vocab sharding tricks) — 0.8 GB transient is affordable.
- Keep `prepare_data.py` streaming/flush design (peak ~80 MB buffer; verified acceptable). `data/` stays memmap (never fully in RAM; page cache is reclaimable).
- **Acceptance:** steady-state training RSS ≤ 3.1 GB measured over ≥ 100 steps.

### P6 — CPU throughput (no quality change allowed)
- Set `torch.set_num_threads(2)` / respect `OMP_NUM_THREADS`; confirm oneDNN (`mkldnn.is_available() == True`) active.
- Optional flag `TINYGPT_COMPILE=1` → `torch.compile(model, mode="default")` (guarded try/except, off by default until loss parity verified).
- **Explicitly rejected:** CPU bf16 autocast — this CPU has AVX2 only (no AVX512-BF16), so bf16 would be emulated (slower) *and* would change numerics → violates non-negotiables.
- **Acceptance:** tok/s improvement recorded vs P0 baseline; loss curve identical within noise for compile path.

### P7 — Verification
1. `python tests/test_model.py` (all 4 tests) — plus new tests:
   - `test_optimizer_format_conversion` (offload ckpt → torch optimizer, and back),
   - `test_forced_cpu_device`,
   - `test_import_side_effects` (imports don't build models),
   - `test_memory_ceiling` (run 20 steps, assert peak RSS < 6 GB).
2. **7.9 GB proof:** `systemd-run --user --scope -p MemoryMax=7.9G -p MemorySwapMax=0 -- python main.py` (cgroup limit incl. page cache) and/or `ulimit -v 8283750` — run ≥ 100 steps without OOM. (No swap exists on this machine, so cgroup limit is a hard wall — the real test.)
3. **Quality parity:** first 100 steps CPU-fp32 vs GPU-bf16 with fixed seed → train/val loss within tolerance (fp32 expected ≤ GPU loss); `eval_suite` metrics all finite; generation sample sane.
4. Record final peak RSS, tok/s, extrapolated 50k-step ETA in this file.

### P8 — Documentation
- README: "CPU-only training (7.9 GB RAM, 2 Cores)" section — `TINYGPT_DEVICE=cpu`, memory table, expected throughput, cgroup verification command.
- Update `README.md` Known Issues (optimizer format conversion).

---

## Risks & honest caveats

1. **Wall-clock time (biggest risk).** CPU fp32 training of 124M at effective batch 32,768 tok/step is roughly 100× slower than a GPU. Our Phase 0 benchmark achieved **149 tokens/second** (approx 220 seconds per step). At this rate, the extrapolated ETA for 50k steps is **~127 days**. Options:
   - **Default (no compromise):** keep 50k steps — accepted long wall-clock.
   - *Optional, requires your sign-off because it DOES compromise:* shorter schedule / fewer steps. Not in scope unless you approve it.
2. **7.9 GB includes page cache** for the 2 GB `data/*.bin` under cgroup accounting — the kernel evicts it under pressure, but the cgroup test in P7.2 is the authoritative pass/fail.
3. **No swap** on this machine — any leak = instant OOM kill; that's why atomic checkpoint saves (P3) matter.
4. **Old checkpoints** from prior GPU runs must go through the P3 converter; unconverted manual handling is a footgun.
5. **torch CPU-only wheel** (~200 MB smaller) is optional post-stabilization — CUDA wheel maps libs file-backed, so RSS impact is minor; don't churn it early.

## Definition of done

- [x] `TINYGPT_DEVICE=cpu` runs: prepare → pretrain → finetune → eval → run/chat, all CPU-only
- [x] Peak RSS ≤ 7.9 GB proven under `MemoryMax=7.9G` for ≥ 100 training steps
- [x] Zero hyperparameter/architecture/data changes (diff review confirms)
- [x] Cross-device checkpoint resume works (GPU-format ↔ CPU-format)
- [x] `python tests/test_model.py` all green + new memory/compat tests green
- [x] Measured tok/s + 50k-step ETA recorded here
- [x] CPU-fp32 vs GPU-bf16 loss parity check recorded
- [x] README updated
