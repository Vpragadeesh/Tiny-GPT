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
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from pathlib import Path

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION (must match main.py)
# ═════════════════════════════════════════════════════════════════════════════

BLOCK_SIZE = 128
EMBED_DIM = 768
NUM_HEADS = 12
NUM_LAYERS = 12
NUM_EXPERTS = 8
TOP_K = 2
FFN_DIM = EMBED_DIM * 4
DROPOUT = 0.1
CHECKPOINT_DIR = "checkpoints"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

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
    global BLOCK_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, NUM_EXPERTS, FFN_DIM, vocab_size

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

    router_key = "blocks.0.moe.router.weight"
    if router_key in state_dict:
        NUM_EXPERTS = state_dict[router_key].shape[0]

    ffn_key = "blocks.0.moe.experts.0.w1.weight"
    if ffn_key in state_dict:
        FFN_DIM = state_dict[ffn_key].shape[0]
    else:
        FFN_DIM = EMBED_DIM * 4

    NUM_HEADS = _infer_num_heads(EMBED_DIM)


# ═════════════════════════════════════════════════════════════════════════════
# 2. MODEL ARCHITECTURE (minimal — see main.py for full details)
# ═════════════════════════════════════════════════════════════════════════════


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_heads = NUM_HEADS
        self.head_dim = EMBED_DIM // NUM_HEADS
        self.qkv = nn.Linear(EMBED_DIM, 3 * EMBED_DIM, bias=False)
        self.proj = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        self.attn_drop = nn.Dropout(DROPOUT)
        self.proj_drop = nn.Dropout(DROPOUT)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)).view(
                1, 1, BLOCK_SIZE, BLOCK_SIZE
            ),
        )

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        att = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att.float(), dim=-1).to(x.dtype)
        att = self.attn_drop(att)

        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj_drop(self.proj(out))


class ExpertFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(EMBED_DIM, FFN_DIM)
        self.w2 = nn.Linear(FFN_DIM, EMBED_DIM)
        self.act = nn.GELU()
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        return self.drop(self.w2(self.act(self.w1(x))))


class MoELayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = nn.Linear(EMBED_DIM, NUM_EXPERTS, bias=False)
        self.experts = nn.ModuleList([ExpertFFN() for _ in range(NUM_EXPERTS)])

    def forward(self, x):
        B, T, C = x.shape
        flat = x.reshape(-1, C)
        N = flat.shape[0]

        logits = self.router(flat)
        probs = F.softmax(logits.float(), dim=-1)

        top_w, top_i = torch.topk(probs, TOP_K, dim=-1)
        top_w = (top_w / top_w.sum(dim=-1, keepdim=True)).to(x.dtype)

        out = torch.zeros_like(flat)
        for i, expert in enumerate(self.experts):
            mask = (top_i == i).any(dim=-1)
            if not mask.any():
                continue
            tokens = flat[mask]
            e_out = expert(tokens)
            match = (top_i[mask] == i).to(x.dtype)
            weights = (top_w[mask] * match).sum(-1, keepdim=True)
            out[mask] += weights * e_out

        return out.reshape(B, T, C)


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(EMBED_DIM)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(EMBED_DIM)
        self.moe = MoELayer()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.moe(self.ln2(x))
        return x


class MoEGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, EMBED_DIM)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, EMBED_DIM)
        self.drop = nn.Dropout(DROPOUT)
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(NUM_LAYERS)])
        self.ln_f = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif "bias" in name:
                nn.init.zeros_(p)
        scale = (2 * NUM_LAYERS) ** -0.5
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=0.02 * scale)
            for expert in block.moe.experts:
                nn.init.normal_(expert.w2.weight, mean=0.0, std=0.02 * scale)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.drop(
            self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        )

        for block in self.blocks:
            x = block(x)

        logits = self.head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens=200,
        temperature=0.8,
        top_k=None,
        top_p=0.9,
    ):
        """
        Generate text from a prompt.

        Args:
            prompt: Starting text
            max_new_tokens: How many tokens to generate
            temperature: Higher = more random (0.5-1.5 typical)
            top_k: Keep only top-k most likely tokens (None = disabled)
            top_p: Nucleus sampling threshold (0.9 typical)
        """
        self.eval()
        ids = torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)

        for _ in range(max_new_tokens):
            ctx = ids[:, -BLOCK_SIZE:]
            with torch.amp.autocast(
                "cuda", dtype=torch.bfloat16, enabled=(DTYPE == torch.bfloat16)
            ):
                logits, _ = self(ctx)
            logits = logits[:, -1, :].float() / temperature

            # Top-K filtering
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")

            # Top-P (nucleus) filtering
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

        self.train()
        return decode(ids[0].tolist())


# ═════════════════════════════════════════════════════════════════════════════
# 3. LOAD MODEL FROM CHECKPOINT
# ═════════════════════════════════════════════════════════════════════════════


def load_model(checkpoint_path=None):
    """Load the trained model from checkpoint."""
    if checkpoint_path is None:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "best.pt")

    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found at: {checkpoint_path}")
        print(f"[ERROR] Have you run 'python main.py' yet?")
        sys.exit(1)

    print(f"Loading model from {checkpoint_path} ...", end=" ", flush=True)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    apply_model_config_from_state_dict(ckpt["model_state"])

    model = MoEGPT()
    model = model.to(dtype=DTYPE, device=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print("✓")
    print(f"  Device: {DEVICE.upper()}")
    print(f"  Dtype: {DTYPE}")
    print(
        f"  Model: block={BLOCK_SIZE}, emb={EMBED_DIM}, heads={NUM_HEADS}, "
        f"layers={NUM_LAYERS}, experts={NUM_EXPERTS}, ffn={FFN_DIM}"
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
            output = model.generate(
                user_input,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        print(output)
        print()

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
            output = model.generate(
                prompt,
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

    # Load model
    model = load_model(args.checkpoint)

    # Dispatch to appropriate mode
    if args.prompt:
        # Single prompt
        print(f"Prompt: {args.prompt}\n")
        with torch.no_grad():
            output = model.generate(
                args.prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
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
