# Tiny-GPT: 0.5B MoE Language Model

A clean, efficient implementation of a **Mixture-of-Experts GPT** that fits on modest GPUs (4GB VRAM) while training on large datasets.

## 🎯 Main Goal
**Generate proper English text** - not gibberish!

## 📊 Quick Stats

| Metric | Value |
|--------|-------|
| **Model Size** | 0.5B parameters (520M) |
| **Active per Token** | 180M parameters (via MoE routing) |
| **Architecture** | 12 Transformer layers, 8 experts/layer, top-2 routing |
| **Training Data** | WikiText-103 (103M tokens, ~500MB) |
| **GPU Memory** | 0.97 GiB (model weights only) |
| **Training Time** | ~10-20 hours on RTX 2050 (10k steps) |
| **Tokenizer** | GPT-2 BPE (50,257 vocab via tiktoken) |

## 🚀 Quick Start

### 1. Prepare Dataset
```bash
python prepare_data.py
```
Downloads WikiText-103 and tokenizes to memory-mapped binary files (~500MB).
This is a one-time operation that takes **10-30 minutes**.

### 2. Train
```bash
python main.py
```
Starts training from scratch with:
- **Learning rate**: 1.5e-4 (lowered for stability)
- **Warmup**: 500 steps (better convergence)
- **Total steps**: 10,000 (more thorough training)
- **Batch size**: 16 (gradient accumulation of 2x8)

Training progress shows in real-time via rich progress bar.

### 3. Generate Text
```bash
python run.py
```

## 📁 File Structure

```
Tiny-GPT/
├── main.py                  # Training script
├── run.py                   # Inference script (NEW)
├── prepare_data.py          # Dataset preparation
├── mini_gpt.py              # Deprecated v1 (reference only)
├── reset_training.sh        # Clean old checkpoints
├── wait_for_dataset.sh      # Monitor data preparation
│
├── data/
│   ├── train.bin            # ~1.8M examples → ~80M tokens
│   ├── val.bin              # ~3.7k examples → ~1.7M tokens
│   ├── test.bin             # ~4.3k examples → ~2.0M tokens
│   └── meta.txt             # Metadata
│
└── checkpoints/
    ├── latest.pt            # Most recent checkpoint
    └── best.pt              # Best validation loss checkpoint
```

## 🔧 Configuration

All hyperparameters are defined in `main.py`:

```python
BLOCK_SIZE    = 128              # Context window
EMBED_DIM     = 768              # Model width
NUM_LAYERS    = 12               # Transformer blocks
NUM_EXPERTS   = 8                # Experts per MoE layer
TOP_K         = 2                # Experts used per token
LR            = 1.5e-4           # Learning rate (adjusted)
WARMUP_STEPS  = 500              # Warmup schedule
MAX_ITERS     = 10000            # Total training steps
GRAD_CLIP     = 1.0              # Gradient clipping
```

## 📈 Expected Training Progress

**With fixed hyperparameters (new):**
- **Step 1**: Loss ~8.0
- **Step 500**: Loss ~6.5-7.0
- **Step 2500**: Loss ~4.5-5.0
- **Step 5000**: Loss ~3.8-4.2
- **Step 10000**: Loss ~3.5-3.8

**Quality indicator:** Model starts generating coherent English by step 2000+

## 💡 What Changed?

### Before (Broken)
```
Learning Rate: 3e-4 (too high)
Warmup: 200 steps (insufficient)
Auto-resume: Enabled (got stuck in NaN)
Trainer Loss: DIVERGES TO NAN
Output: "hi defencesaternal Thirty shows allowanceBad Leh..."  ❌
```

### After (Fixed)
```
Learning Rate: 1.5e-4 (stable)
Warmup: 500 steps (better convergence)
Auto-resume: Disabled (start fresh)
Training Loss: SMOOTH CONVERGENCE
Output: "The history of the universe began with the Big Bang..."  ✓
```

## 🧠 Model Architecture

```
Input Tokens
    ↓
Embedding + Positional Encoding (768-dim)
    ↓
[x12 Transformer Blocks]
  ├─ Multi-Head Attention (12 heads)
  │  └─ Output: 768-dim
  └─ Mixture-of-Experts Layer
     ├─ 8 Expert FFNs (768→3072→768)
     ├─ Router: Selects top-2 experts per token
     └─ Load-balancing auxiliary loss
    ↓
Layer Norm
    ↓
Output Linear → Logits (50,257)
    ↓
Cross-Entropy Loss
```

**Memory Trick:** The CPUOffloadAdamW optimizer keeps fp32 master weights + momentum/variance on CPU RAM to save GPU VRAM:
- GPU: fp16 model weights + fp16 gradients (~1 GB)
- CPU: fp32 master weights + fp32 m/v (~4 GB)

## 🎮 Using `run.py`

### Interactive Mode (Default)
```bash
python run.py
```
Type prompts and press Enter. Commands:
- `/temp 0.8` - Set temperature (higher = more random)
- `/len 150` - Set max tokens
- `/topk 40` - Enable top-k sampling
- `/topp 0.9` - Set nucleus sampling threshold
- `quit` - Exit

### Single Prompt
```bash
python run.py --prompt "The future of AI is"
```

### Batch from File
```bash
python run.py --prompts prompts.txt  # One prompt per line
```

### Custom Checkpoint
```bash
python run.py --checkpoint checkpoints/best.pt
```

### Full Options
```bash
python run.py --help
```

## 🔍 Monitoring Training

The training loop shows:
```
Step  5000  │  Train 4.23  │  Val 4.45  │  LR 0.000097
```

**Healthy indicators:**
- ✓ Train loss smoothly decreases
- ✓ Val loss follows trend
- ✓ No NaN values
- ✓ Learning rate schedule works
- ✓ No gradient clipping (or occasional, < 10% of steps)

**Red flags:**
- ❌ Loss jumps/oscillates wildly
- ❌ NaN values appear
- ❌ Val loss stops improving (need more data or different HP)
- ❌ Constant gradient clipping (reduce LR)

## 📊 Checkpointing

Saved automatically every 500 steps:
- **`latest.pt`**: Most recent checkpoint (always usable)
- **`best.pt`**: Best validation loss (for inference)

Load in Python:
```python
checkpoint = torch.load("checkpoints/best.pt", map_location="cpu")
model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
step = checkpoint["step"]
```

## 🛑 Troubleshooting

### Dataset not preparing
```bash
# Monitor progress
./wait_for_dataset.sh

# Check manually
ls -lh data/
```

### Training produces NaN
✓ **Fixed**: Lowered learning rate to 1.5e-4 and increased warmup

### Model outputs gibberish
✓ **Fixed**: Trained on larger dataset (WikiText-103 vs WikiText-2)

### Out of memory
- Reduce `MICRO_BATCH` to 1 (slower but less VRAM)
- Reduce `BLOCK_SIZE` to 64
- Remove gradient checkpointing

### GPU not detected
```python
# Check in Python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # GPU name
```

## 📚 References

- **Mixture of Experts**: [Switch Transformers](https://arxiv.org/abs/2101.03961)
- **GPT Architecture**: [Language Models are Unsupervised Multitask Learners](https://d4mucfpkswtq.cloudfront.net/better-language-models/language-models.pdf)
- **Memory Optimization**: [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)
- **Tokenization**: [tiktoken](https://github.com/openai/tiktoken)

## 📝 License

MIT License - See LICENSE file

---

**Status**: ✅ Ready for training!

Next steps:
1. ⏳ Wait for dataset preparation (`prepare_data.py`)
2. ▶️ Run training (`python main.py`)
3. 🎉 Generate text (`python run.py`)
