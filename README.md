# Tiny-GPT: 124M Dense Transformer

A minimal, efficient GPT-2 Small implementation optimized for training on modest GPUs (4GB VRAM) and quantization research.

## Architecture

- **Model**: 124M parameter dense transformer (no MoE)
- **Config**: 12 layers, 768-dim, 12 heads, 512 context window
- **Tokenizer**: GPT-2 BPE (50,257 vocab via tiktoken)
- **Training Data**: FineWeb-Edu (9.75B tokens) + Alpaca instruction fine-tuning
- **Hardware**: RTX 2050 (4GB VRAM) via CPUOffloadAdamW

## Quick Start

### 1. Prepare Dataset
```bash
python prepare_data.py
```
Streams and tokenizes FineWeb-Edu from HuggingFace to memory-mapped .bin files.

### 2. Pre-train
```bash
python main.py
```
Trains 124M dense GPT-2 from scratch with:
- **Learning rate**: 1.5e-4 with cosine decay
- **Warmup**: 500 steps
- **Total steps**: 50,000
- **Effective batch**: 64 (MICRO_BATCH=4, GRAD_ACCUM=16)

### 3. Fine-tune (optional)
```bash
python finetune.py
```
Fine-tunes on Alpaca instruction data. Must have pre-trained `best.pt` first.

### 4. Generate Text
```bash
python run.py --prompt "Hello"
python chat.py  # interactive chat (fine-tuned model)
```

### 5. DeepSpeed Training (multi-GPU)
```bash
bash train_deepspeed.sh
```

## CPU-Only Training (7.9 GB RAM / 2 Cores)

The entire pipeline supports fully deterministic, CPU-only execution while staying strictly under a 7.9 GB RAM budget. To force CPU training or inference, use:

```bash
TINYGPT_DEVICE=cpu python main.py
```

**Memory Breakdown (Peak RSS: ~2.5 GB)**
- Model Weights (FP32): ~500 MB
- Optimizer States (AdamW): ~1.0 GB
- Gradients (FP32): ~500 MB
- Python/Torch overhead: ~500 MB

*Note*: CPU execution processes ~149 tokens/second on 2 cores. At an effective batch size of 64 (32,768 tokens per step), reaching 50,000 steps will take approximately 127 days. You can safely stop and resume using the robust checkpointing system.

**Verification Command (Linux):**
```bash
systemd-run --user --scope -p MemoryMax=7.9G -p MemorySwapMax=0 -- TINYGPT_DEVICE=cpu python main.py
```

## Memory Optimization

The CPUOffloadAdamW optimizer keeps fp32 master weights + momentum/variance on CPU RAM to fit on 4GB VRAM:
- **GPU**: bf16 model weights + gradients (~1 GB)
- **CPU**: fp32 master weights + fp32 m/v (~6 GB)

## Evaluation

The `eval_suite.py` module tracks:
- Train/val loss & perplexity
- Token accuracy
- Generation diversity ratio
- Generation samples

Integrated into the training loop -- visible in progress bar output.

## File Structure

```
Tiny-GPT/
├── main.py                  # Pre-training script
├── main_deepspeed.py        # DeepSpeed ZeRO-2 variant
├── finetune.py              # Instruction fine-tuning
├── run.py                   # Inference (interactive/single/batch/HF Hub)
├── chat.py                  # Interactive chat (fine-tuned)
├── prepare_data.py          # FineWeb-Edu streaming -> .bin files
├── prepare_chat_data.py     # Alpaca -> .bin files
├── push_to_hf.py            # Upload checkpoints to HuggingFace Hub
├── eval_suite.py            # Comprehensive evaluation metrics
├── benchmark_attention.py   # Softmax vs linear attention benchmark
├── model.py                 # TinyGPT model definition
├── tinygpt/                 # Modular package
│   ├── attention/           # CausalSelfAttention, LinearAttention
│   ├── layers/              # FeedForward, TransformerBlock
│   ├── training/            # CPUOffloadAdamW, scheduler, checkpoint
│   └── generation/          # (pending)
├── data/                    # Pre-training binary data
├── instruction_data/        # Fine-tuning binary data (Alpaca)
└── tests/                   # Unit tests
    └── test_model.py
```

## Dependencies

```bash
pip install torch tiktoken numpy rich datasets tqdm
# Optional:
pip install deepspeed          # Multi-GPU training
pip install huggingface_hub    # Upload/download checkpoints
```

## Known Issues

- **Weight tying**: `head.weight` is tied to `tok_emb.weight`. Checkpoints from older training runs with separate `head.weight` are handled in `load_checkpoint()`.
- **Checkpoint format conversion**: GPU checkpoints (using `CPUOffloadAdamW` state) and CPU checkpoints (using PyTorch `AdamW` state) have structurally different optimizer dictionaries. `load_checkpoint()` seamlessly converts them, but custom loading scripts must use the `convert_optimizer_state()` helper.
- **chat.py**: Requires a fine-tuned checkpoint. Falls back to pre-trained if not available. Checkpoint dimension mismatches may occur with different model configs.

## License

MIT License
