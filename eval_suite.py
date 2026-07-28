"""
Comprehensive evaluation suite for Tiny-GPT.
Tracks loss, perplexity, token accuracy, and generation diversity.
"""

import torch
import tiktoken
from tinygpt.training.evaluation import estimate_loss


def eval_suite(model, get_batch, eval_iters=100, device="cuda", verbose=False):
    """Run comprehensive evaluation and return metrics dict.

    Args:
        model: TinyGPT model instance
        get_batch: data loading function get_batch(split)
        eval_iters: number of batches for loss estimation
        device: torch device string
        verbose: if True, print results

    Returns:
        dict with keys: train_loss, val_loss, train_ppl, val_ppl,
                        token_accuracy, generation_sample, diversity_ratio
    """
    enc = tiktoken.get_encoding("gpt2")

    # -- Loss / Perplexity --
    losses = estimate_loss(model, get_batch, eval_iters)
    train_ppl = torch.exp(torch.tensor(losses["train"])).item()
    val_ppl = torch.exp(torch.tensor(losses["val"])).item()

    # -- Token Accuracy --
    model.eval()
    x, y = get_batch("val")
    with torch.no_grad():
        logits, _ = model(x)
        preds = logits.argmax(dim=-1)
        correct = (preds == y).float().sum().item()
        total = y.numel()
        token_accuracy = correct / total

    # -- Generation Sample --
    prompt_ids = torch.tensor([[enc.eot_token]], dtype=torch.long, device=device)
    generated = ""
    with torch.no_grad():
        idx = prompt_ids
        for _ in range(50):
            logits, _ = model(idx[:, -512:])
            logits = logits[:, -1, :] / 0.8
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
            if nxt.item() == enc.eot_token:
                break
    generated = enc.decode(idx[0].tolist())

    # -- Diversity (repetition ratio) --
    words = generated.split()
    diversity_ratio = len(set(words)) / max(len(words), 1)

    model.train()

    results = {
        "train_loss": losses["train"],
        "val_loss": losses["val"],
        "train_ppl": train_ppl,
        "val_ppl": val_ppl,
        "token_accuracy": token_accuracy,
        "generation_sample": generated[:200],
        "diversity_ratio": diversity_ratio,
    }

    if verbose:
        print(f"  Train loss: {results['train_loss']:.4f}  |  PPL: {results['train_ppl']:.2f}")
        print(f"  Val loss:   {results['val_loss']:.4f}  |  PPL: {results['val_ppl']:.2f}")
        print(f"  Token accuracy: {results['token_accuracy']:.4f}")
        print(f"  Diversity ratio: {results['diversity_ratio']:.3f}")
        print(f"  Sample: {results['generation_sample'][:100]}...")

    return results


if __name__ == "__main__":
    # Quick smoke test
    from model import TinyGPT
    from tinygpt.training import load_checkpoint
    from main import get_batch, DEVICE

    model = TinyGPT(50257, 512, 768, 12, 12, 3072, 0.1).to(DEVICE)
    try:
        import os
        ckpt_path = os.path.join("checkpoints", "best.pt")
        if os.path.exists(ckpt_path):
            step, val_loss = load_checkpoint(ckpt_path, model, None)
            print(f"Loaded checkpoint (step {step}, val_loss {val_loss:.4f})")
    except Exception as e:
        print(f"No checkpoint loaded: {e}")

    results = eval_suite(model, get_batch, eval_iters=10, device=DEVICE, verbose=True)
