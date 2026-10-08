"""
run.py – Inference script for MoE-GPT
========================================
Run the trained model anytime to generate text.

Usage:
    python run.py                  # Interactive mode
    python run.py --prompt "text"  # Generate from prompt
    python run.py --file data.txt  # Generate continuations from file

No training — just inference from the best checkpoint.
"""

import os
import sys
import argparse
import gc
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
import tiktoken
from model import CausalSelfAttention, FeedForward, TransformerBlock, TinyGPT
from tinygpt.device import resolve_device, resolve_dtype, autocast_ctx

# ═════════════════════════════════════════════════════════════════════════════
# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION (must match main.py) — 95M config
# ════════════════════════════════════════════════════════════════════════════
BLOCK_SIZE = 512
EMBED_DIM = 768
NUM_HEADS = 12
NUM_LAYERS = 10
FFN_DIM = EMBED_DIM * 4
DROPOUT = 0.0
CHECKPOINT_DIR = "checkpoints"

DEVICE = resolve_device()
DTYPE = resolve_dtype(DEVICE)

# ═════════════════════════════════════════════════════════════════════════════
# 1. TOKENISER – GPT-2 BPE
# ═════════════════════════════════════════════════════════════════════════════

enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab  # 50,257


def encode(text: str) -> list:
    return enc.encode_ordinary(text)


def decode(ids: list) -> str:
    return enc.decode(ids)


def _infer_num_heads(embed_dim: int) -> int:
    """Infer a reasonable attention head count from embedding size."""
    for h in (16, 12, 8, 6, 4, 2, 1):
        if embed_dim % h == 0:
            return h
    return 1


def apply_model_config_from_state_dict(state_dict: dict):
    """Update global model hyperparameters to match checkpoint tensors."""
    global BLOCK_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, FFN_DIM, vocab_size

    if "tok_emb.weight" not in state_dict or "pos_emb.weight" not in state_dict:
        return

    vocab_size = state_dict["tok_emb.weight"].shape[0]
    EMBED_DIM = state_dict["tok_emb.weight"].shape[1]
    BLOCK_SIZE = state_dict["pos_emb.weight"].shape[0]

    layer_ids = []
    for k in state_dict.keys():
        if k.startswith("blocks."):
            parts = k.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                layer_ids.append(int(parts[1]))
    if layer_ids:
        NUM_LAYERS = max(layer_ids) + 1

    ffn_key = "blocks.0.ffn.w1.weight"
    if ffn_key in state_dict:
        FFN_DIM = state_dict[ffn_key].shape[0]
    else:
        FFN_DIM = EMBED_DIM * 4

    if EMBED_DIM == 768:
        NUM_HEADS = 12
    else:
        NUM_HEADS = _infer_num_heads(EMBED_DIM)


def _get_model_state_from_checkpoint(ckpt: dict) -> dict:
    """Support both training checkpoint formats used in this repo."""
    if "model_state" in ckpt:
        return ckpt["model_state"]
    if "model" in ckpt:
        return ckpt["model"]
    raise KeyError("Checkpoint does not contain 'model_state' or 'model'")


def resolve_checkpoint_path(
    checkpoint_path=None,
    hf_repo=None,
    hf_filename="best.pt",
    hf_revision=None,
    hf_token=None,
):
    """Resolve a local checkpoint path, optionally downloading from HF Hub."""
    if hf_repo:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            print("[ERROR] huggingface_hub is required for --hf-repo")
            print("[ERROR] Install it with: pip install huggingface_hub")
            sys.exit(1)

        cache_dir = Path("hf_cache") / "hub"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return hf_hub_download(
            repo_id=hf_repo,
            filename=hf_filename,
            revision=hf_revision,
            token=hf_token,
            cache_dir=str(cache_dir),
        )

    if checkpoint_path is None:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "best.pt")
    return checkpoint_path


# ═════════════════════════════════════════════════════════════════════════════
# 2. MODEL ARCHITECTURE — see model.py for TinyGPT, TransformerBlock, etc.
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate(model, prompt: str, max_new_tokens=200, temperature=0.8,
             top_k=None, top_p=0.9):
    model.eval()
    ids = torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)

    for _ in range(max_new_tokens):
        ctx = ids[:, -BLOCK_SIZE:]
        with autocast_ctx(DEVICE, DTYPE):
            logits, _ = model(ctx)
        logits = logits[:, -1, :].float() / temperature

        if top_k is not None:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float("-inf")

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumsum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumsum_probs > top_p
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[:, indices_to_remove] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)
        ids = torch.cat([ids, nxt], dim=1)

    model.train()
    return decode(ids[0].tolist())


# ═════════════════════════════════════════════════════════════════════════════
# 3. LOAD MODEL FROM CHECKPOINT
# ═════════════════════════════════════════════════════════════════════════════


def load_model(
    checkpoint_path=None,
    hf_repo=None,
    hf_filename="best.pt",
    hf_revision=None,
    hf_token=None,
):
    """Load the trained model from checkpoint."""
    checkpoint_path = resolve_checkpoint_path(
        checkpoint_path=checkpoint_path,
        hf_repo=hf_repo,
        hf_filename=hf_filename,
        hf_revision=hf_revision,
        hf_token=hf_token,
    )

    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found at: {checkpoint_path}")
        print(f"[ERROR] Have you run 'python main.py' yet?")
        sys.exit(1)

    print(f"Loading model from {checkpoint_path} ...", end=" ", flush=True)
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    model_state = _get_model_state_from_checkpoint(ckpt)
    
    # Drop optimizer state to free up memory before building model
    if "optimizer" in ckpt:
        del ckpt["optimizer"]
    del ckpt
    gc.collect()
    
    # Use fixed 10-layer config (95M) instead of inferring from checkpoint
    global NUM_LAYERS
    NUM_LAYERS = 10
    # Skip apply_model_config_from_state_dict to preserve fixed config

    model = TinyGPT(vocab_size, BLOCK_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS,
                    FFN_DIM, DROPOUT, use_manual_attention=True)
    model = model.to(dtype=DTYPE, device=DEVICE)
    model.load_state_dict(model_state, strict=False)
    
    del model_state
    gc.collect()
    
    model.eval()

    print("✓")
    print(f"  Device: {DEVICE.upper()}")
    print(f"  Dtype: {DTYPE}")
    print(
        f"  Model: block={BLOCK_SIZE}, emb={EMBED_DIM}, heads={NUM_HEADS}, "
        f"layers={NUM_LAYERS}, ffn={FFN_DIM}"
    )
    print()

    return model


# ═════════════════════════════════════════════════════════════════════════════
# 4. INTERACTIVE & BATCH INFERENCE
# ═════════════════════════════════════════════════════════════════════════════


def interactive_mode(model):
    """Interactive text generation."""
    print("=" * 70)
    print("Interactive Mode – Type 'quit' to exit")
    print("=" * 70)
    print()
    print("Commands:")
    print("  quit          – Exit")
    print("  /temp 0.7     – Set temperature (default 0.8)")
    print("  /len 100      – Set max tokens (default 200)")
    print("  /topk 40      – Set top-k (default None = disabled)")
    print("  /topp 0.9     – Set top-p (default 0.9)")
    print()

    model.eval()
    temperature = 0.8
    max_tokens = 200
    top_k = None
    top_p = 0.9

    while True:
        try:
            user_input = input("Prompt > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            break

        # Handle commands
        if user_input.startswith("/"):
            parts = user_input.split()
            if len(parts) == 2:
                cmd, val = parts[0][1:], parts[1]
                try:
                    if cmd == "temp":
                        temperature = float(val)
                        print(f"Temperature set to {temperature}")
                    elif cmd == "len":
                        max_tokens = int(val)
                        print(f"Max tokens set to {max_tokens}")
                    elif cmd == "topk":
                        top_k = int(val)
                        print(f"Top-k set to {top_k}")
                    elif cmd == "topp":
                        top_p = float(val)
                        print(f"Top-p set to {top_p}")
                except ValueError:
                    print(f"Invalid value for {cmd}")
            continue

        print()
        with torch.no_grad():
            output = generate(
                model, user_input,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        print(output)
        print()

    model.train()
    print("\nGoodbye!")


def batch_generation(model, prompts, max_tokens=200, temperature=0.8):
    """Generate from a list of prompts."""
    print("=" * 70)
    print("Batch Generation")
    print("=" * 70)
    print()

    with torch.no_grad():
        for i, prompt in enumerate(prompts, 1):
            print(f"[{i}/{len(prompts)}] Prompt: {prompt}")
            output = generate(
                model, prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )
            print(f"Output: {output}\n")


# ═════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ═════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Generate text using trained MoE-GPT model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                          # Interactive mode
  python run.py --prompt "Hello world"   # Generate from prompt
  python run.py --prompts file.txt       # Batch from file (one per line)
  python run.py --checkpoint custom.pt   # Use custom checkpoint
    python run.py --hf-repo user/Tiny-GPT  # Load from Hugging Face Hub
        """,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Single prompt to generate from",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        help="File with prompts (one per line) for batch generation",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint (default: checkpoints/best.pt)",
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=None,
        help="Hugging Face repo id (e.g. user/Tiny-GPT). If set, download checkpoint from HF Hub.",
    )
    parser.add_argument(
        "--hf-filename",
        type=str,
        default="best.pt",
        help="Filename inside HF repo (default: best.pt)",
    )
    parser.add_argument(
        "--hf-revision",
        type=str,
        default=None,
        help="HF branch/tag/commit to download from",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HF token for private repos (or use HF_TOKEN env var)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Max tokens to generate (default: 200)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling (default: disabled)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p/nucleus sampling (default: 0.9)",
    )

    args = parser.parse_args()

    if args.hf_repo and args.checkpoint:
        print("[ERROR] Use either --checkpoint or --hf-repo, not both.")
        sys.exit(1)

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    # Load model
    model = load_model(
        checkpoint_path=args.checkpoint,
        hf_repo=args.hf_repo,
        hf_filename=args.hf_filename,
        hf_revision=args.hf_revision,
        hf_token=hf_token,
    )

    # Dispatch to appropriate mode
    if args.prompt:
        # Single prompt
        print(f"Prompt: {args.prompt}\n")
        model.eval()
        with torch.no_grad():
            output = generate(
                model, args.prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
        model.train()
        print(output)

    elif args.prompts:
        # Batch from file
        if not os.path.exists(args.prompts):
            print(f"[ERROR] File not found: {args.prompts}")
            sys.exit(1)
        with open(args.prompts) as f:
            prompts = [line.strip() for line in f if line.strip()]
        batch_generation(model, prompts, args.max_tokens, args.temperature)

    else:
        # Interactive mode
        interactive_mode(model)


if __name__ == "__main__":
    main()
