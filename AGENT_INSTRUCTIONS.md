# Agent Instructions: Tiny-GPT Improvement & Quantization Setup

**Project Goal**: Fix critical bugs, optimize training, and prepare for TurboQuant (W8A8 + ternary quantization) integration.

**Target Model**: 124M dense GPT-2 at `/home/pragadeesh/Tiny-GPT`  
**Hardware**: RTX 2050 (4GB VRAM), 16GB RAM, Arch Linux  
**Outcome**: Runnable, testable baseline + quantization-ready architecture

---

## PHASE 1: CRASH FIXES & STABILITY (2–4 hours)

### Task 1.1: Fix chat.py import crash
**File**: `/home/pragadeesh/Tiny-GPT/chat.py`

**Problem**: Line 20 calls `TinyGPT()` with zero arguments, but the constructor requires 7 positional params: `vocab_size, block_size, embed_dim, num_heads, num_layers, ffn_dim, dropout`.

**Action**:
1. Read `model.py` to find the default config (should be 50257 vocab, 512 block_size, 768 embed_dim, 12 heads, 12 layers, 3072 ffn_dim, 0.1 dropout)
2. Replace line 20 from:
   ```python
   model = TinyGPT().to(dtype=DTYPE, device=DEVICE)
   ```
   to:
   ```python
   model = TinyGPT(
       vocab_size=50257, block_size=512, embed_dim=768,
       num_heads=12, num_layers=12, ffn_dim=3072, dropout=0.1,
       attention_cls=CausalSelfAttention, use_manual_attention=False
   ).to(dtype=DTYPE, device=DEVICE)
   ```
3. Add `if __name__ == "__main__":` guard around the main execution (line ~80+) so the file can be imported without side effects
4. Test: `python -c "import chat; print('OK')"`

**Success criteria**: File imports without error; interactive mode can start (no GPU required for import test).

---

### Task 1.2: Fix finetune.py import guard
**File**: `/home/pragadeesh/Tiny-GPT/finetune.py`

**Problem**: No `if __name__` guard. The entire training loop executes on import, causing side effects (data loading, model init, training start).

**Action**:
1. Wrap all code after the config section (starting from `# Load data` comment) in `if __name__ == "__main__":`
2. Ensure proper indentation
3. Test: `python -c "import finetune; print('OK')"` — should not load data or start training
4. Test: `python finetune.py` — should still work normally

**Success criteria**: File can be imported without triggering training; running directly still trains.

---

### Task 1.3: Fix checkpoint weight tying
**File**: `/home/pragadeesh/Tiny-GPT/tinygpt/training/checkpoint.py`

**Problem**: When loading checkpoints, if the state dict has a separate `head.weight` entry (from older saves), it breaks weight tying. The model expects `head.weight` to be tied to `tok_emb.weight`.

**Action**:
1. In `load_checkpoint()` function, after loading the state dict, add:
   ```python
   # Handle weight tying: remove separate head.weight if present
   if 'head.weight' in state_dict:
       del state_dict['head.weight']
   # Also remove any 'model.head.weight' key
   state_dict = {k: v for k, v in state_dict.items() if k != 'model.head.weight'}
   ```
2. This ensures the head always uses the tied embedding weight
3. Test by saving a checkpoint, modifying it to add separate head weights, then loading it

**Success criteria**: Checkpoints load without key mismatch errors; model runs inference correctly after load.

---

### Task 1.4: Unify checkpoint keys across main.py and main_deepspeed.py
**Files**: `/home/pragadeesh/Tiny-GPT/main.py` and `main_deepspeed.py`

**Problem**: `main.py` saves with key `"model"`, `main_deepspeed.py` loads/saves with `"model_state"`. This breaks resuming training across scripts.

**Action**:
1. Read both files' checkpoint save/load sections
2. In `main.py`, around line 180–190 (in `save_checkpoint` call):
   - Ensure the key is `"model_state"` (not `"model"`)
3. In `main_deepspeed.py`, ensure it also uses `"model_state"`
4. Update `tinygpt/training/checkpoint.py` if needed to use consistent naming
5. Test: Train 10 steps on main.py, save best.pt, load in main_deepspeed.py (or vice versa)

**Success criteria**: Checkpoints are compatible across both training scripts.

---

### Task 1.5: Remove linear_attention.py duplicate
**File**: `/home/pragadeesh/Tiny-GPT/linear_attention.py`

**Problem**: Identical copy of `tinygpt/attention/linear.py`. Duplicates cause confusion and maintenance debt.

**Action**:
1. Delete `/home/pragadeesh/Tiny-GPT/linear_attention.py`
2. Check `benchmark_attention.py` — it imports from `linear_attention`. Update it to:
   ```python
   from tinygpt.attention.linear import LinearAttention
   ```
3. Test: `python benchmark_attention.py` runs without import errors

**Success criteria**: No duplicate files; benchmark still runs.

---

## PHASE 2: DATA & TRAINING OPTIMIZATION (3–5 days)

### Task 2.1: Increase batch size and implement gradient accumulation
**Files**: `main.py`, `finetune.py`

**Current state**: Batch size is likely 4–8 (too small for effective learning).

**Action**:
1. In `main.py`, around line 30–40 (config section), change:
   ```python
   BATCH_SIZE = 32  # up from 4 or 8
   GRAD_ACCUM_STEPS = 2  # gradient accumulation every 2 steps
   EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRAD_ACCUM_STEPS  # 64 effective
   ```
2. Update training loop to accumulate gradients:
   ```python
   loss_accum = 0
   for step in range(max_iters):
       x, y = get_batch('train')
       logits, loss = model(x, y)
       loss = loss / GRAD_ACCUM_STEPS
       loss.backward()
       loss_accum += loss.item()
       
       if (step + 1) % GRAD_ACCUM_STEPS == 0:
           torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
           optimizer.step()
           optimizer.zero_grad()
           # log loss_accum every log interval
   ```
3. Do the same in `finetune.py`
4. Test: Run 50 steps, verify loss decreases more consistently (less noisy gradient updates)

**Success criteria**: Training is more stable; loss curve is less jagged; no OOM errors.

---

### Task 2.2: Fix learning rate schedule in training loop
**Files**: `main.py`, `finetune.py`

**Current state**: LR may be fixed or poorly scheduled.

**Action**:
1. Ensure `tinygpt/training/scheduler.py` has `get_lr(step, lr, warmup_steps, max_iters)` function
2. In the training loop (main.py), around line 140–160, update every step:
   ```python
   current_lr = get_lr(step, lr=LR, warmup_steps=WARMUP_STEPS, max_iters=MAX_ITERS)
   for param_group in optimizer.param_groups:
       param_group['lr'] = current_lr
   ```
3. For finetune.py, use a 10× lower peak LR:
   ```python
   FINETUNE_LR = LR / 10  # 1.5e-5 instead of 1.5e-4
   ```
4. Test: Log LR at each step, verify it warmups linearly then decays cosine

**Success criteria**: LR schedule matches expected curve; final LR is ~10% of peak.

---

### Task 2.3: Add data quality checks
**File**: `prepare_data.py`

**Action**:
1. After tokenizing, add sanity checks:
   ```python
   # Check for extreme sequences (all same token, too many special tokens)
   if len(set(tokens)) < 50:
       print(f"Warning: Example {i} has low diversity, skipping")
       continue
   ```
2. Log min/max sequence length, vocab coverage, EOT token frequency
3. Print dataset stats before and after filtering (e.g., "Removed 2% of examples due to low quality")

**Success criteria**: Dataset prep is transparent; you can spot if data is corrupted.

---

### Task 2.4: Implement evaluation tracking
**File**: Create new file `/home/pragadeesh/Tiny-GPT/eval_suite.py`

**Purpose**: Track multiple metrics per eval cycle.

**Action**:
```python
import torch
from tinygpt.training.evaluation import estimate_loss

def eval_suite(model, get_batch, eval_iters=100):
    """Comprehensive evaluation: perplexity, token accuracy, generation coherence."""
    
    # Perplexity
    losses = estimate_loss(model, get_batch, eval_iters)
    train_ppl = torch.exp(torch.tensor(losses['train']))
    val_ppl = torch.exp(torch.tensor(losses['val']))
    
    # Generation coherence (simple heuristic)
    model.eval()
    with torch.no_grad():
        prompt_ids = torch.tensor([[50256]])  # BOS token
        generated = model.generate(prompt_ids, max_new_tokens=50, temperature=0.7, top_k=40)
        text = tiktoken.get_encoding('gpt2').decode(generated[0].tolist())
        
        # Check for repetition
        words = text.split()
        unique_words = len(set(words))
        repetition_ratio = unique_words / max(len(words), 1)
    
    return {
        'train_ppl': train_ppl.item(),
        'val_ppl': val_ppl.item(),
        'train_loss': losses['train'],
        'val_loss': losses['val'],
        'generation_sample': text,
        'diversity_ratio': repetition_ratio
    }

if __name__ == "__main__":
    # Test it
    from model import TinyGPT
    model = TinyGPT(50257, 512, 768, 12, 12, 3072, 0.1).cuda()
    # ... load checkpoint, run eval_suite(model, get_batch)
```

2. Integrate into main.py eval loop to track metrics every EVAL_EVERY steps
3. Log to CSV or JSON for plotting

**Success criteria**: Metrics recorded; can plot loss, perplexity, diversity over time.

---

## PHASE 3: QUANTIZATION PREPARATION (2–3 weeks)

### Task 3.1: Add W8A8 fake quantization wrapper
**File**: Create `/home/pragadeesh/Tiny-GPT/tinygpt/quantization.py`

**Purpose**: Simulates W8A8 (weights int8, activations int8) during training so the model learns to be quantized.

**Action**:
```python
import torch
import torch.nn as nn

class FakeQuantize(nn.Module):
    """Simulates quantization: scale, round to int8, then dequantize back to float."""
    
    def __init__(self, num_bits=8):
        super().__init__()
        self.num_bits = num_bits
        self.scale = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x):
        # Symmetric quantization: scale [-128, 127]
        q_min, q_max = -(2 ** (self.num_bits - 1)), 2 ** (self.num_bits - 1) - 1
        
        # Determine scale (max absolute value)
        x_abs_max = x.abs().max()
        scale = x_abs_max / (q_max - q_min) * 2  # symmetric range
        
        # Quantize
        x_q = torch.clamp(torch.round(x / scale), q_min, q_max)
        
        # Dequantize
        x_dq = x_q * scale
        return x_dq

class QuantizableLinear(nn.Linear):
    """Linear layer that can enable fake quantization."""
    
    def __init__(self, in_features, out_features, bias=True, fake_quant=False):
        super().__init__(in_features, out_features, bias)
        self.fake_quant = fake_quant
        if fake_quant:
            self.weight_quant = FakeQuantize(num_bits=8)
            self.act_quant = FakeQuantize(num_bits=8)
    
    def forward(self, x):
        if self.fake_quant:
            w = self.weight_quant(self.weight)
            x = self.act_quant(x)
        else:
            w = self.weight
        
        return torch.nn.functional.linear(x, w, self.bias)

# Usage in training:
# model = TinyGPT(..., use_fake_quant=True)  # enable W8A8 during training
# Train for first 80% of steps normally
# For last 20%, enable fake_quant=True to learn quantization
```

2. Document where to hook this into `model.py` or training loop
3. Test: Run 10 steps with fake_quant=False, then 10 with fake_quant=True, verify model still learns

**Success criteria**: Fake quantization reduces loss increase to <0.5% over unquantized baseline.

---

### Task 3.2: Integrate TurboQuant PolarQuantizer
**File**: `/home/pragadeesh/Tiny-GPT/tinygpt/quantization_turboquant.py`

**Purpose**: Use your existing PolarQuantizer W8A8 hybrid approach with BitNet b1.58 ternary targets.

**Action**:
1. Import your PolarQuantizer class from `/home/pragadeesh/tmp/1.58-bit-LLM-Trubo-Quant/`
2. Create a wrapper:
   ```python
   from polar_quant import PolarQuantizer
   
   class TurboQuantLayer(nn.Module):
       """Wraps a Linear layer with TurboQuant W8A8 + ternary readiness."""
       
       def __init__(self, linear_layer):
           super().__init__()
           self.linear = linear_layer
           self.polar_quant = PolarQuantizer(
               in_features=linear_layer.in_features,
               out_features=linear_layer.out_features
           )
       
       def forward(self, x, use_quantization=False):
           if use_quantization:
               w_q = self.polar_quant.quantize(self.linear.weight)
               return torch.nn.functional.linear(x, w_q, self.linear.bias)
           else:
               return self.linear(x)
   ```
3. Document how to swap Linear → TurboQuantLayer in model.py
4. Test: Verify inference works with and without quantization

**Success criteria**: Inference runs with TurboQuant; latency/memory is measurable.

---

### Task 3.3: Benchmark quantization impact
**File**: `/home/pragadeesh/Tiny-GPT/bench_quantization.py`

**Action**:
```python
import torch
import time
from model import TinyGPT

def benchmark_quantization():
    """Compare: float32, W8A8 fake, ternary inference."""
    
    model_fp32 = TinyGPT(50257, 512, 768, 12, 12, 3072, 0.1).cuda()
    model_fp32.eval()
    
    prompt = torch.randint(0, 50257, (1, 512)).cuda()
    
    # FP32 baseline
    with torch.no_grad():
        start = time.time()
        for _ in range(100):
            _ = model_fp32(prompt)
        fp32_time = (time.time() - start) / 100
    
    # W8A8 (fake quant)
    # ... enable fake quant, benchmark
    
    # Ternary (inference only, post-training conversion)
    # ... convert weights to ternary, benchmark
    
    print(f"FP32: {fp32_time:.3f}s/iter")
    print(f"W8A8 speedup: {fp32_time / w8a8_time:.2f}x")
    print(f"Ternary speedup: {fp32_time / ternary_time:.2f}x")

if __name__ == "__main__":
    benchmark_quantization()
```

**Success criteria**: Report speedup and memory savings for each quantization method.

---

## PHASE 4: VALIDATION & TESTING (1 week)

### Task 4.1: Build unit tests
**File**: `/home/pragadeesh/Tiny-GPT/tests/test_model.py`

**Action**:
```python
import pytest
import torch
from model import TinyGPT
from tinygpt.training.checkpoint import save_checkpoint, load_checkpoint

def test_model_forward():
    model = TinyGPT(50257, 512, 768, 12, 12, 3072, 0.1)
    x = torch.randint(0, 50257, (2, 128))
    logits, loss = model(x, x)
    assert logits.shape == (2, 128, 50257)
    assert loss.item() > 0

def test_checkpoint_save_load():
    model = TinyGPT(50257, 512, 768, 12, 12, 3072, 0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    save_checkpoint(1, model, optimizer, 4.5, 5.1, 'test.pt')
    step, val_loss = load_checkpoint('test.pt', model, optimizer)
    
    assert step == 1
    assert val_loss == 5.1

def test_chat_init():
    # Verify chat.py imports and initializes without error
    from chat import model
    assert model is not None

if __name__ == "__main__":
    pytest.main([__file__, '-v'])
```

2. Create `/home/pragadeesh/Tiny-GPT/tests/__init__.py` (empty file)
3. Run: `pytest tests/ -v`

**Success criteria**: All tests pass.

---

### Task 4.2: Create training baseline
**Action**:
1. Train for 1000 steps with fixed hyperparameters (no LR schedule yet):
   - Batch size: 32
   - LR: 1e-4 (constant)
   - Eval every 100 steps
2. Save baseline loss curve as `baseline_loss.json`
3. Run again with improved hyperparameters (LR schedule, warmup, higher batch):
   - Same 1000 steps
   - Save as `improved_loss.json`
4. Plot both curves; verify improvement is measurable (5–10% lower val loss)

**Success criteria**: Clear measurable improvement; reproducible results.

---

## PHASE 5: DOCUMENTATION & HANDOFF (2 days)

### Task 5.1: Update README.md
**Action**:
1. Replace stale MoE description with accurate dense GPT info:
   ```markdown
   # Tiny-GPT: 124M Dense Transformer
   
   A minimal, efficient GPT-2 implementation optimized for ternary quantization research.
   
   ## Architecture
   - **Model**: 124M dense transformer (no MoE)
   - **Config**: 12 layers, 768-dim, 12 heads, 512 context
   - **Training**: FineWeb-Edu (9.75B tokens) + Alpaca instruction fine-tuning
   - **Hardware**: RTX 2050 (4GB VRAM) via CPUOffloadAdamW
   
   ## Quick Start
   ```bash
   # Pre-training
   python main.py
   
   # Fine-tuning
   python finetune.py
   
   # Inference + chat
   python run.py --prompt "Hello"
   python chat.py  # interactive
   
   # Quantization-aware training (W8A8)
   python main.py --fake_quant --enable_ternary_warmup
   
   # Benchmarking
   python bench_quantization.py
   ```
   ```
2. Add section: "Known Issues" (weight tying, checkpoint keys—now fixed)
3. Add section: "Quantization" with TurboQuant integration notes

**Success criteria**: README is accurate and helpful for future users.

---

### Task 5.2: Create IMPROVEMENTS.md
**Action**:
```markdown
# Improvements Applied

## Phase 1: Critical Fixes
- [x] Fixed chat.py crash (TinyGPT() → constructor with args)
- [x] Added __main__ guard to finetune.py
- [x] Fixed weight tying in checkpoint loading
- [x] Unified checkpoint keys (main.py ↔ main_deepspeed.py)
- [x] Removed duplicate linear_attention.py

## Phase 2: Training Optimization
- [x] Batch size: 4 → 32 (gradient accumulation ×2)
- [x] LR schedule: fixed → warmup + cosine decay
- [x] Eval suite: perplexity, diversity, generation quality
- [x] Data quality checks in prepare_data.py

## Phase 3: Quantization Readiness
- [x] W8A8 fake quantization wrapper
- [x] TurboQuant integration (PolarQuantizer)
- [x] Quantization benchmarking suite
- [x] Ternary warmup hooks

## Results
- Final train loss: [TBD after full run]
- Final val loss: [TBD after full run]
- Perplexity @ convergence: [TBD]
- W8A8 inference speedup: [TBD]
- Ternary inference speedup: [TBD]
```

**Success criteria**: Clear record of what was improved and why.

---

## EXECUTION CHECKLIST

- [ ] Phase 1 (Crashes): All 5 tasks complete, test passing
- [ ] Phase 2 (Training): Batch size up, LR schedule active, eval suite tracking
- [ ] Phase 3 (Quantization): W8A8 fake quant works, TurboQuant integrated, benchmarks run
- [ ] Phase 4 (Validation): Unit tests pass, baseline vs improved comparison done
- [ ] Phase 5 (Documentation): README updated, IMPROVEMENTS.md written

---

## SUCCESS METRICS

**After all phases:**

1. ✅ Code runs without crashes (chat.py, finetune.py, main.py all import cleanly)
2. ✅ Training is reproducible (same hyperparams → same loss curve ±0.1%)
3. ✅ Eval tracks 5+ metrics (loss, perplexity, diversity, generation quality, throughput)
4. ✅ Quantization is measurable (W8A8: <1% accuracy loss, 2–3× speedup)
5. ✅ Tests pass (>80% coverage of core modules)
6. ✅ Documentation is current (README, IMPROVEMENTS.md, inline comments)

---

## NOTES FOR AGENT

- **Priority**: Fix crashes first (Phase 1) before optimization (Phase 2+)
- **Testing**: After each task, run a quick sanity check (import test, 10-step training test)
- **Rollback**: If a change breaks training, revert immediately and document the failure
- **Performance**: Prefer correctness over speed; measure before optimizing
- **Commit**: After each phase, commit to git with clear message (e.g., "Fix: chat.py init crash")

