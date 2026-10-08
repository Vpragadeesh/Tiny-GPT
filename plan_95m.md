# Implementation Plan: 95M Parameter Config for 2c/8GB CPU

## Target Config
| Param | Value | Change |
|---|---|---|
| `BLOCK_SIZE` | 512 | same |
| `MICRO_BATCH` | 8 | ↑ from 4 |
| `GRAD_ACCUM` | 8 | ↓ from 16 (eff batch = 64) |
| `EMBED_DIM` | 768 | same |
| `NUM_HEADS` | 12 | same |
| `NUM_LAYERS` | 10 | ↓ from 12 |
| `FFN_DIM` | 3072 | same (768×4) |
| `LR` | 2.5e-4 | ↑ from 1.5e-4 |
| `MAX_ITERS` | 50_000 | fix from 10 |
| `USE_ACTIVATION_CHECKPOINT` | True | same |

**Params:** ~95M  
**Peak RAM:** ~6.5 GB (of 7 GB budget)  
**Est. 2c time:** ~30-35 hours for 50k steps

---

## Files to Modify

### 1. `main.py` (lines 97-115)
- Update all hyperparams above
- Fix `MAX_ITERS = 50_000`
- Update `EFFECTIVE_BATCH` comment

### 2. `finetune.py` (lines 80-96)
- Match `main.py` exactly (BLOCK_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, FFN_DIM)
- Update MICRO_BATCH=8, GRAD_ACCUM=8, LR=2.5e-4
- Keep fine-tune specific: LR=2e-5, MAX_ITERS=10_000

### 3. `run.py` (lines 30-36)
- Update inference defaults to match

### 4. `chat.py` (lines 16, 24-27)
- Update `BLOCK_SIZE`, model constructor args

---

## Verification Steps
1. `python -m pytest tests/ -v` — all tests pass
2. `TINYGPT_DEVICE=cpu python -c "from main import build_training_ctx; build_training_ctx()"` — model builds, ~95M params
3. `python tools/monitor.py python -c "import os; os.environ['TINYGPT_DEVICE']='cpu'; from main import build_training_ctx; import torch; m,o,s,b=build_training_ctx(); x=torch.randint(0,50257,(8,512)); y=torch.randint(0,50257,(8,512)); _,l=m(x,y); l.backward(); o.step(); o.zero_grad(); print('step ok')"` — 1 step, peak RSS ≤ 7 GB
4. `TINYGPT_DEVICE=cpu python chat.py` (quick test) — loads and generates