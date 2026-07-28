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
- **Checkpoint key**: Unified to `model_state` across all training scripts.
- **chat.py**: Requires a fine-tuned checkpoint. Falls back to pre-trained if not available. Checkpoint dimension mismatches may occur with different model configs.

## License

MIT License
