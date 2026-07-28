# Tiny-GPT Architecture Report

Reverse-engineered from source code only. No README trust.

---

## 1. Dependency Graph

```
chat.py
  └── main.py (import: TinyGPT, BLOCK_SIZE, DEVICE, DTYPE, vocab_size)
        ├── torch, torch.nn, torch.nn.functional
        ├── torch.utils.checkpoint (grad_checkpoint)
        ├── tiktoken
        ├── numpy
        └── rich (progress, console, table)

finetune.py
  ├── torch, torch.nn, torch.nn.functional
  ├── torch.utils.checkpoint (grad_checkpoint)
  ├── tiktoken
  ├── numpy
  └── rich (progress, console)

run.py
  ├── torch, torch.nn, torch.nn.functional
  ├── torch.utils.checkpoint (grad_checkpoint)
  ├── tiktoken
  └── huggingface_hub (hf_hub_download) [optional]

prepare_data.py
  ├── tiktoken
  ├── numpy
  ├── datasets (load_dataset)
  └── tqdm

prepare_chat_data.py
  ├── tiktoken
  ├── numpy
  ├── datasets (load_dataset)
  └── tqdm

push_to_hf.py
  └── huggingface_hub (HfApi, upload_file)

main_deepspeed.py
  ├── torch, torch.nn, torch.nn.functional
  ├── torch.utils.checkpoint (grad_checkpoint)
  ├── tiktoken
  ├── deepspeed
  ├── numpy
  └── rich (progress, console)
```

---

## 2. Call Graph

### main.py

```
__main__ block
  ├── TinyGPT()
  │     ├── nn.Embedding() x2 (tok_emb, pos_emb)
  │     ├── TransformerBlock() x12
  │     │     ├── CausalSelfAttention()
  │     │     │     ├── nn.Linear() x2 (qkv, proj)
  │     │     │     └── forward()
  │     │     │           └── F.scaled_dot_product_attention()
  │     │     ├── FeedForward()
  │     │     │     ├── nn.Linear() x2 (w1, w2)
  │     │     │     └── forward()
  │     │     └── forward()
  │     ├── nn.LayerNorm()
  │     ├── nn.Linear() (head)
  │     └── _init_weights()
  ├── CPUOffloadAdamW(model.parameters())
  ├── load_checkpoint() [if exists]
  ├── get_batch() [in estimate_loss()]
  │     └── np.memmap → torch.from_numpy()
  ├── estimate_loss()
  │     └── model(x, y) x EVAL_ITERS
  ├── Training Loop
  │     ├── get_lr(step)
  │     ├── get_batch("train") x GRAD_ACCUM
  │     ├── model(x, y)
  │     ├── loss.backward()
  │     ├── clip_grad_norm_()
  │     ├── optimizer.step()
  │     ├── estimate_loss() [every EVAL_EVERY]
  │     └── save_checkpoint() [latest.pt, best.pt]
  ├── model.generate() [3 prompts]
  └── Interactive Mode
        └── model.generate()
```

### finetune.py

```
Top-level (no __main__ guard)
  ├── TinyGPT() [duplicated class definitions]
  ├── CPUOffloadAdamW()
  ├── Load pretrained checkpoint (best.pt)
  ├── Training Loop
  │     ├── get_lr(step)
  │     ├── get_batch("train") x GRAD_ACCUM
  │     ├── model(x, y)
  │     ├── loss.backward()
  │     ├── clip_grad_norm_()
  │     ├── optimizer.step()
  │     └── save_checkpoint() [finetune_latest.pt, finetune_best.pt]
  └── Test Evaluation
```

### chat.py

```
Top-level (no __main__ guard)
  ├── contextlib.redirect_stdout
  │     └── from main import TinyGPT, ...
  ├── TinyGPT().to(DTYPE, DEVICE)
  ├── torch.load() [finetune_best.pt → best.pt → latest.pt]
  ├── model.load_state_dict()
  ├── model.eval()
  └── Chat Loop
        ├── enc.encode_ordinary(chat_history)
        ├── Sliding window truncation [if > BLOCK_SIZE-100]
        ├── model(ctx) → logits
        ├── Top-p nucleus sampling
        ├── torch.multinomial()
        └── enc.decode()
```

### run.py

```
main()
  ├── argparse
  ├── resolve_checkpoint_path()
  │     └── hf_hub_download() [if HF repo]
  ├── apply_model_config_from_state_dict()
  ├── load_model()
  │     ├── TinyGPT()
  │     ├── torch.load()
  │     ├── _get_model_state_from_checkpoint()
  │     └── model.load_state_dict()
  ├── interactive_mode() | batch_generation()
  │     └── model.generate()
  └── [or single prompt via generate()]
```

---

## 3. Module Graph

```
┌─────────────────────────────────────────────────────────────┐
│                      Tiny-GPT Project                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ prepare_data │    │prepare_chat  │    │  push_to_hf  │  │
│  │    .py       │    │  _data.py    │    │    .py       │  │
│  │              │    │              │    │              │  │
│  │ Input:       │    │ Input:       │    │ Input:       │  │
│  │ FineWeb-Edu  │    │ Alpaca       │    │ checkpoints/ │  │
│  │ (streaming)  │    │ (HF dataset) │    │              │  │
│  │              │    │              │    │ Output:      │  │
│  │ Output:      │    │ Output:      │    │ HF Hub       │  │
│  │ data/*.bin   │    │instruction_  │    │              │  │
│  │              │    │  data/*.bin  │    │              │  │
│  └──────┬───────┘    └──────┬───────┘    └──────────────┘  │
│         │                   │                               │
│         ▼                   ▼                               │
│  ┌──────────────┐    ┌──────────────┐                      │
│  │   main.py    │    │ finetune.py  │                      │
│  │              │    │              │                      │
│  │ Data: data/  │    │ Data:        │                      │
│  │ Arch: 124M   │    │ instruction_ │                      │
│  │ Optim:       │    │   data/      │                      │
│  │ CPUOffload   │    │ Arch: 124M   │                      │
│  │   AdamW      │    │ Optim:       │                      │
│  │              │    │ CPUOffload   │                      │
│  │ Output:      │    │   AdamW      │                      │
│  │ checkpoints/ │    │              │                      │
│  │  best.pt     │    │ Output:      │                      │
│  │  latest.pt   │    │ checkpoints/ │                      │
│  └──────┬───────┘    │  finetune_   │                      │
│         │            │  best.pt     │                      │
│         │            │  finetune_   │                      │
│         │            │  latest.pt   │                      │
│         │            └──────┬───────┘                      │
│         │                   │                               │
│         ▼                   ▼                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   chat.py    │    │   run.py     │    │main_deep-    │  │
│  │              │    │              │    │  speed.py    │  │
│  │ Imports from │    │ Standalone   │    │              │  │
│  │ main.py      │    │ model defs   │    │ Standalone   │  │
│  │              │    │              │    │ model defs   │  │
│  │ Loads:       │    │ Loads:       │    │              │  │
│  │ finetune_    │    │ any .pt or   │    │ Uses:        │  │
│  │  best.pt     │    │ HF Hub       │    │ DeepSpeed    │  │
│  │ → best.pt    │    │              │    │ ZeRO-2       │  │
│  │ → latest.pt  │    │ Modes:       │    │              │  │
│  │              │    │ interactive  │    │ Config:      │  │
│  │ Output:      │    │ single       │    │ ds_config    │  │
│  │ Chat REPL    │    │ batch        │    │   .json      │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Training Pipeline

### 4a. Pre-training (main.py)

```
1. Load Data
   └── np.memmap("data/{train,val,test}.bin") → uint16 tokens

2. Tokenize (already done by prepare_data.py)
   └── tiktoken GPT-2 BPE (50,257 vocab)

3. Model Init
   └── TinyGPT() → 124M params
       ├── tok_emb: Embedding(50257, 768)
       ├── pos_emb: Embedding(512, 768)
       ├── blocks: 12x TransformerBlock
       │     ├── ln1: LayerNorm(768)
       │     ├── attn: CausalSelfAttention
       │     │     ├── qkv: Linear(768, 2304, bias=False)
       │     │     └── proj: Linear(768, 768, bias=False)
       │     ├── ln2: LayerNorm(768)
       │     └── ffn: FeedForward
       │           ├── w1: Linear(768, 3072)
       │           └── w2: Linear(3072, 768)
       ├── ln_f: LayerNorm(768)
       └── head: Linear(768, 50257, bias=False)
           └── weight tied to tok_emb.weight

4. Optimizer Init
   └── CPUOffloadAdamW(model.parameters(), lr=1.5e-4)
       ├── GPU: bf16 model weights + bf16 gradients
       └── CPU: fp32 master weights + fp32 momentum + fp32 variance

5. Checkpoint Resume
   └── If latest.pt exists → load model + optimizer state

6. Training Loop (500k steps)
   ├── LR Schedule
   │     ├── Warmup: 0 → 1.5e-4 over 500 steps
   │     └── Decay: cosine to 1.5e-5 over 500k steps
   ├── Per Step:
   │     ├── optimizer.zero_grad()
   │     ├── for _ in range(8):  # GRAD_ACCUM
   │     │     ├── get_batch("train") → (x, y)  # MICRO_BATCH=2
   │     │     ├── autocast(bf16)
   │     │     ├── model(x, y) → logits, loss
   │     │     └── (loss/8).backward()
   │     ├── clip_grad_norm_(1.0)
   │     └── optimizer.step()
   │           └── CPUOffloadAdamW:
   │                 ├── Copy grad GPU→CPU (fp16→fp32)
   │                 ├── Weight decay on master
   │                 ├── Update m, v (fp32)
   │                 ├── Bias-corrected update
   │                 └── Copy master CPU→GPU (fp32→fp16)
   └── Every 1000 steps:
         ├── estimate_loss() → {train: float, val: float}
         ├── save_checkpoint(latest.pt)
         └── save_checkpoint(best.pt) [if val improved]
```

### 4b. Fine-tuning (finetune.py)

```
1. Load Instruction Data
   └── np.memmap("instruction_data/{train,val,test}.bin")

2. Model Init
   └── TinyGPT() [same 124M architecture]

3. Load Pre-trained Checkpoint
   └── checkpoints/best.pt → model.load_state_dict()

4. Optimizer Init
   └── CPUOffloadAdamW(model.parameters(), lr=2e-5)
       └── 10x lower LR to preserve pre-trained weights

5. Training Loop (10k steps)
   ├── Same structure as pre-training
   ├── Saves to: finetune_latest.pt, finetune_best.pt
   └── Does NOT overwrite best.pt
```

---

## 5. Inference Pipeline

### 5a. Interactive Chat (chat.py)

```
1. Load Model
   ├── contextlib.redirect_stdout (silence main.py imports)
   ├── from main import TinyGPT, BLOCK_SIZE, DEVICE, DTYPE
   ├── TinyGPT().to(DTYPE, DEVICE)
   └── Load checkpoint:
         ├── Try: checkpoints/finetune_best.pt
         ├── Try: checkpoints/best.pt
         └── Try: checkpoints/latest.pt

2. Initialize Chat
   ├── SYSTEM_PROMPT = "System: You are a helpful assistant.\n"
   └── chat_history = SYSTEM_PROMPT

3. Per User Input:
   ├── Append: "User: {input}\nAssistant:"
   ├── Encode: enc.encode_ordinary(chat_history)
   ├── Sliding Window:
   │     └── If tokens > BLOCK_SIZE-100:
   │           ├── Keep system prompt tokens
   │           └── Truncate middle of history
   ├── Generate (max 100 tokens):
   │     └── for _ in range(100):
   │           ├── ctx = ids[:, -512:]
   │           ├── autocast(bf16)
   │           ├── model(ctx) → logits
   │           ├── temperature = 0.7
   │           ├── Top-p (nucleus) filtering, p=0.9
   │           ├── torch.multinomial(probs, 1)
   │           └── Stop on EOT or newline
   ├── Decode: enc.decode(generated_tokens)
   ├── Print: "Bot: {response}"
   └── Append: " {response}\n" to chat_history
```

### 5b. Standalone Inference (run.py)

```
1. Resolve Checkpoint
   ├── Local path, or
   └── HuggingFace Hub download

2. Auto-detect Model Config
   └── apply_model_config_from_state_dict()
         ├── Read tok_emb.weight.shape → vocab_size, EMBED_DIM
         ├── Read blocks[0].attn.qkv.weight.shape → NUM_HEADS
         └── Count blocks → NUM_LAYERS

3. Load Model
   └── TinyGPT() → load_state_dict()

4. Modes:
   ├── Interactive: REPL with /temp, /len, /topk, /topp commands
   ├── Single: generate from one prompt
   └── Batch: generate from file of prompts
```

---

## 6. Model Pipeline

```
Input: token indices [B, T] (int64)
  │
  ▼
┌─────────────────────────────────────────┐
│ tok_emb: Embedding(vocab_size, 768)     │  → [B, T, 768]
│ pos_emb: Embedding(512, 768)            │  → [T, 768]
│ x = dropout(tok_emb + pos_emb)          │  → [B, T, 768]
└─────────────────────────────────────────┘
  │
  ▼  × 12 TransformerBlocks
┌─────────────────────────────────────────┐
│ TransformerBlock:                        │
│   ├── residual = x                       │
│   ├── x = LayerNorm(768)(x)             │
│   ├── x = CausalSelfAttention(x)        │
│   │     ├── qkv = Linear(768, 2304)(x)  │  → [B, T, 2304]
│   │     ├── reshape → [B, T, 12, 64]    │
│   │     ├── permute → [B, 12, T, 64]    │
│   │     ├── SDPA(q, k, v, is_causal)    │  → [B, 12, T, 64]
│   │     ├── reshape → [B, T, 768]       │
│   │     └── proj: Linear(768, 768)      │  → [B, T, 768]
│   ├── x = residual + x                   │
│   ├── residual = x                       │
│   ├── x = LayerNorm(768)(x)             │
│   ├── x = FeedForward(x)                │
│   │     ├── w1: Linear(768, 3072)       │  → [B, T, 3072]
│   │     ├── GELU                         │
│   │     ├── w2: Linear(3072, 768)       │  → [B, T, 768]
│   │     └── dropout                     │
│   └── x = residual + x                   │
└─────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────┐
│ x = LayerNorm(768)(x)                   │
│ logits = Linear(768, 50257)(x)          │  → [B, T, 50257]
│ loss = CrossEntropy(logits, targets)     │  → scalar (if targets given)
└─────────────────────────────────────────┘
```

---

## 7. Dataset Pipeline

### 7a. Pre-training Data (prepare_data.py)

```
Source: HuggingFaceFW/fineweb-edu (sample-10BT)
  │
  ▼
Streaming Download (no full download)
  │
  ▼
Per Row:
  ├── extract_text(row) → str
  │     └── Try: text, content, output, row[0]
  ├── encode_text(text) → list[int]
  │     ├── enc.encode_ordinary(text)
  │     └── append(EOT token)
  ├── pick_split(i, total) → "train"|"val"|"test"
  │     └── 98% train, 1% val, 1% test
  └── flush_tokens(fp, buffer) → int
        └── Write uint16 to disk every 2M tokens

Output:
  data/train.bin  (~9.75B tokens)
  data/val.bin    (~100M tokens)
  data/test.bin   (~100M tokens)
  data/meta.txt   (statistics)
```

### 7b. Instruction Data (prepare_chat_data.py)

```
Source: tatsu-lab/alpaca (52K examples)
  │
  ▼
Full Download
  │
  ▼
Per Row:
  ├── format_instruction(row) → str
  │     ├── If input:
  │     │     "System: You are a helpful assistant.\nUser: {instruction}\n{input}\nAssistant: {output}"
  │     └── Else:
  │           "System: You are a helpful assistant.\nUser: {instruction}\nAssistant: {output}"
  ├── enc.encode_ordinary(text)
  └── append(EOT token)

Split: 95% train, 2.5% val, 2.5% test

Output:
  instruction_data/train.bin
  instruction_data/val.bin
  instruction_data/test.bin
```

---

## 8. Class & Function Reference

### Classes

| Class | File | Responsibility |
|-------|------|----------------|
| `CausalSelfAttention` | main.py:126, main_deepspeed.py:144, finetune.py:116, run.py:141 | Fused QKV multi-head self-attention with FlashAttention (SDPA) |
| `FeedForward` | main.py:152, main_deepspeed.py:170, finetune.py:138, run.py:171 | Two-layer FFN: Linear→GELU→Linear→Dropout |
| `TransformerBlock` | main.py:166, main_deepspeed.py:184, finetune.py:150, run.py:183 | Pre-norm transformer block: LN→Attn→Residual→LN→FFN→Residual |
| `TinyGPT` | main.py:182, main_deepspeed.py:200, finetune.py:164, run.py:197 | Full GPT model: embeddings + transformer blocks + LM head |
| `CPUOffloadAdamW` | main.py:274, finetune.py:192 | AdamW with fp32 state on CPU, bf16 on GPU |
| `_Wrap` | main.py:440, finetune.py:319 | Adapter wrapping torch.optim.AdamW to match CPUOffloadAdamW interface |

### Functions

| Function | File | Line | Responsibility |
|----------|------|------|----------------|
| `encode(text)` | main.py, finetune.py, run.py | various | GPT-2 BPE encode |
| `decode(ids)` | main.py, finetune.py, run.py | various | GPT-2 BPE decode |
| `get_batch(split)` | main.py:115, finetune.py:105, main_deepspeed.py:133 | Sample random batch from memmap |
| `save_checkpoint(...)` | main.py:352, finetune.py:248, main_deepspeed.py:322 | Serialize model+optimizer to disk |
| `load_checkpoint(...)` | main.py:362, main_deepspeed.py:338 | Deserialize from disk |
| `get_lr(step)` | main.py:375, finetune.py:261, main_deepspeed.py:352 | Linear warmup → cosine decay LR schedule |
| `estimate_loss()` | main.py:386, finetune.py:272, main_deepspeed.py:372 | Average loss over EVAL_ITERS batches |
| `extract_text(row)` | prepare_data.py:51 | Extract text from dataset row |
| `encode_text(text)` | prepare_data.py:67 | Encode + append EOT |
| `flush_tokens(fp, buf)` | prepare_data.py:73 | Write token buffer to disk |
| `pick_split(i, total)` | prepare_data.py:83 | Assign train/val/test split |
| `format_instruction(row)` | prepare_chat_data.py:21 | Format Alpaca row with markers |
| `apply_model_config_from_state_dict(sd)` | run.py:64 | Auto-detect model dims from checkpoint |
| `_get_model_state_from_checkpoint(ckpt)` | run.py:96 | Extract model state from checkpoint dict |
| `resolve_checkpoint_path(...)` | run.py:105 | Find or download checkpoint |
| `load_model(...)` | run.py:296 | Full model loading pipeline |
| `interactive_mode(model)` | run.py:344 | REPL with /temp, /len commands |
| `batch_generation(model, prompts)` | run.py:414 | Generate from prompt list |
| `main()` | run.py:437, push_to_hf.py:19 | CLI entry points |
| `_strip_orig_mod_prefix(sd)` | main_deepspeed.py:286 | Remove torch.compile prefix |
| `_add_orig_mod_prefix(sd)` | main_deepspeed.py:296 | Add torch.compile prefix |
| `_align_state_dict_for_model(sd, model)` | main_deepspeed.py:306 | Match checkpoint keys to model |
| `get_eta_clock(progress, task_id)` | main_deepspeed.py:359 | ETA display in IST |

---

## 9. Key Discrepancies Found

| Issue | Files | Detail |
|-------|-------|--------|
| Attention implementation | run.py vs others | `run.py` uses manual `q @ k.T` with mask buffer; all others use `F.scaled_dot_product_attention` |
| `finetune.py` missing methods | finetune.py | TinyGPT lacks `_init_weights()` and `generate()` methods |
| No `__main__` guard | finetune.py, chat.py, prepare_chat_data.py | Execute on import — causes side effects |
| Checkpoint key mismatch | main.py vs main_deepspeed.py | main.py uses `"model"` key; main_deepspeed.py uses `"model_state"` |
| Activation checkpointing | finetune.py | `USE_ACTIVATION_CHECKPOINT=True` but TinyGPT.forward() doesn't use it |
| `main_deepspeed.py` model size | main_deepspeed.py | Still 101M (512-dim, 8 layers) — not updated to 124M |
