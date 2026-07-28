# Tiny-GPT Bug & Mistake Report

## BUGS (will crash or produce wrong output)

| # | File:Line | Issue | Status |
|---|---|---|---|
| 1 | `haiku/gpt_model.py:160` | `logits[indices_to_remove]` indexes batch dim (size 1) with vocab indices. Should be `logits[:, indices_to_remove]` | **FIXED** |
| 2 | `haiku/generate.py:73` | `interactive_mode()` hardcodes `top_k=50, top_p=0.9` — ignores `--top_k`/`--top_p` CLI args | Open |
| 3 | `haiku/generate.py:94` | `batch_generation()` doesn't pass `top_k`/`top_p` args — always runs with default `temperature` only | Open |
| 4 | `main.py:252` | Weight tying (`self.head.weight = self.tok_emb.weight`) — checkpoint loading breaks if state dict has separate `head.weight` key | Open |
| 5 | `run.py` vs `main.py` | Run inference top-k default `None`; main.py generation uses `top_k=50` — inconsistent sampling behavior | Open |

## DESIGN MISTAKES

| # | File:Line | Issue |
|---|---|---|
| 6 | `haiku/train.py:20-32` | `TokenDataset.__iter__()` infinite loop; epoch terminates at arbitrary `batch_idx > 5000` — never passes through full dataset once |
| 7 | `haiku/prepare_data.py:18` | Downloads `TinyStories-valid` (validation split) instead of training split — tiny corpus |
| 8 | `haiku/prepare_data.py:38-58` | Fallback mock data = 16 lines repeated 100x — model memorizes, not learns |
| 9 | `haiku/gpt_model.py:14` | `weight_tying = False` defined in config but never read by `GPT` class — dead field |
| 10 | `haiku/train.py:25` | `np.uint32` for memmap — tokens max at 50257, `uint16` suffices. Wastes 2x disk |
| 11 | `main.py:95` vs README | Code says `MAX_ITERS = 100_000`; README says "10k steps" — out of sync |
| 12 | `prepare_data.py:38` | `MAX_EXAMPLES = 1000000` limits dataset; existing `train.bin` has 9.75B tokens from bigger run |
| 13 | `prepare_data.py:83` | Train/val split is 98/1/1 — very small validation set, high variance in eval metrics |
| 14 | `kaggle_train.py` | Full duplicate of `main.py` model + training logic — maintenance burden |
| 15 | `run.py:65` | `apply_model_config_from_state_dict()` auto-detects dims from tensors — fragile if tensor names change |
| 16 | `main.py:472` | NaN detection uses `_c.get("val_loss") != _c.get("val_loss")` — correct but obscure vs `math.isnan()` |

## THINGS THAT WORK

- **MoE-GPT 0.5B** trains to convergence on 4GB VRAM GPU via custom `CPUOffloadAdamW` — novel engineering
- **Vanilla GPT (haiku)** loads checkpoint and generates coherent English via CLI/interactive/batch modes
- **Training loop**: BF16 autocast, gradient accumulation, cosine LR schedule, GradScaler, gradient clipping — all correct
- **Data pipeline**: Streaming FineWeb-Edu from HF, GPT-2 BPE tokenization, memory-mapped storage — memory-efficient
- **Checkpointing**: Best/latest/epoch saves work correctly with full state (model + optimizer + step)
- **Inference**: Temperature, top-k, top-p sampling all implemented; interactive mode supports live parameter tuning
- **HuggingFace integration**: `push_to_hf.py` uploads checkpoints; `run.py` downloads from HF Hub — deployable
