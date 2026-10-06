import os
import sys
import io
import contextlib
import torch
import torch.nn.functional as F
import tiktoken
from tinygpt.device import autocast_ctx
from tinygpt.training.checkpoint import load_model_weights

from model import TinyGPT, CausalSelfAttention
from tinygpt.device import resolve_device, resolve_dtype

DEVICE = resolve_device()
DTYPE = resolve_dtype(DEVICE)
BLOCK_SIZE = 512
vocab_size = 50257

# ═════════════════════════════════════════════════════════════════════════════
# LOAD MODEL & CHECKPOINT
# ═════════════════════════════════════════════════════════════════════════════

def build_chat_model():
    model = TinyGPT(
        vocab_size=50257, block_size=512, embed_dim=768,
        num_heads=12, num_layers=12, ffn_dim=3072, dropout=0.1,
        attention_cls=CausalSelfAttention, use_manual_attention=False
    ).to(dtype=DTYPE, device=DEVICE)

    # Resolve checkpoint path — prefer fine-tuned, fallback to pre-trained
    ckpt_path = os.path.join("checkpoints", "finetune_best.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join("checkpoints", "best.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join("checkpoints", "latest.pt")

    if os.path.exists(ckpt_path):
        step, _, _ = load_model_weights(ckpt_path, model)
        print(f"Loaded checkpoint: {ckpt_path} (step {step})")
    else:
        print("No checkpoint found — using random weights")

    model.eval()
    return model

enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token

# ═════════════════════════════════════════════════════════════════════════════
# CHAT INTERFACE
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Alpaca Assistant ready! Type 'quit' to exit.")
    print("-" * 50)

    model = build_chat_model()

    # Initialize with the Alpaca System Prompt
    SYSTEM_PROMPT = "System: You are a helpful assistant.\n"
    chat_history = SYSTEM_PROMPT

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input or user_input.lower() == "quit":
            break

        chat_history += f"User: {user_input}\nAssistant:"

        ids = torch.tensor([enc.encode_ordinary(chat_history)], dtype=torch.long, device=DEVICE)

        # Sliding Window: Ensure we don't exceed the BLOCK_SIZE limit
        if ids.shape[1] > BLOCK_SIZE - 100:  # Leave 100 tokens room for response
            sys_ids = torch.tensor([enc.encode_ordinary(SYSTEM_PROMPT)], dtype=torch.long, device=DEVICE)
            recent_ids = ids[:, -(BLOCK_SIZE - 100 - sys_ids.shape[1]):]
            ids = torch.cat([sys_ids, recent_ids], dim=1)

        generated_tokens = []

        with torch.no_grad():
            for _ in range(100):
                ctx = ids[:, -BLOCK_SIZE:]
                with autocast_ctx(DEVICE, DTYPE):
                    logits, _ = model(ctx)

                # Standard Assistant Sampling Parameters
                temperature = 0.7
                top_p = 0.9

                step_logits = logits[:, -1, :].float() / temperature

                # Top-P (Nucleus) filtering
                sorted_logits, sorted_indices = torch.sort(step_logits, descending=True)
                cumsum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumsum_probs > top_p
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                step_logits[:, indices_to_remove] = float("-inf")

                probs = F.softmax(step_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                if next_token.item() == EOT or enc.decode([next_token.item()]) == "\n":
                    break

                ids = torch.cat([ids, next_token], dim=1)
                generated_tokens.append(next_token.item())

        bot_response = enc.decode(generated_tokens).strip()
        print(f"Bot: {bot_response}")

        chat_history += f" {bot_response}\n"

    print("\nGoodbye!")
