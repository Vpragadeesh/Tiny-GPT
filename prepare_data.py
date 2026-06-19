#!/usr/bin/env python3
"""
prepare_data.py
===============
Build tokenized binary files for training from FineWeb-Edu using streaming.

Outputs:
  data/train.bin
  data/val.bin
  data/test.bin

Dataset:
  HuggingFaceFW/fineweb-edu (streaming)
"""

import os
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm.auto import tqdm

# Local project cache for reproducibility and resume behavior.
os.environ.setdefault("HF_HOME", "./hf_cache")
os.environ.setdefault("HF_DATASETS_CACHE", "./hf_cache/datasets")
os.environ.setdefault("HF_HUB_CACHE", "./hf_cache/hub")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = "./hf_cache"
DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = os.environ.get("DATASET_CONFIG", "sample-10BT")

# Stream only first N rows by default, matching your requested pattern.
MAX_EXAMPLES = int(os.environ.get("MAX_EXAMPLES", "1000000"))

# Deterministic split from one stream: 98% train, 1% val, 1% test.
TRAIN_FRAC = float(os.environ.get("TRAIN_FRAC", "0.98"))
VAL_FRAC = float(os.environ.get("VAL_FRAC", "0.01"))

# Flush chunks to disk to keep RAM bounded.
FLUSH_TOKENS = int(os.environ.get("FLUSH_TOKENS", "2000000"))

enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token


def extract_text(row: dict) -> str:
    """Extract a usable text field across possible FineWeb-Edu schemas."""
    if "text" in row and isinstance(row["text"], str):
        return row["text"].strip()
    if "content" in row and isinstance(row["content"], str):
        return row["content"].strip()

    parts = []
    for key in ("prompt", "question", "instruction", "input", "answer", "response", "output"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())

    return "\n\n".join(parts).strip()


def encode_text(text: str):
    ids = enc.encode_ordinary(text)
    ids.append(EOT)
    return ids


def flush_tokens(fp, buffer_tokens):
    if not buffer_tokens:
        return 0
    arr = np.asarray(buffer_tokens, dtype=np.uint16)
    arr.tofile(fp)
    n = int(arr.size)
    buffer_tokens.clear()
    return n


def pick_split(i: int, total: int) -> str:
    train_cut = int(total * TRAIN_FRAC)
    val_cut = train_cut + int(total * VAL_FRAC)
    if i < train_cut:
        return "train"
    if i < val_cut:
        return "val"
    return "test"


if __name__ == "__main__":
    print("Loading FineWeb-Edu (streaming)...")

    # This follows your requested style while allowing MAX_EXAMPLES override.
    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split="train",
        streaming=True,
        cache_dir=CACHE_DIR,
    )

    out_paths = {
        "train": DATA_DIR / "train.bin",
        "val": DATA_DIR / "val.bin",
        "test": DATA_DIR / "test.bin",
    }

    for p in out_paths.values():
        if p.exists():
            p.unlink()

    buffers = {"train": [], "val": [], "test": []}
    counts_examples = {"train": 0, "val": 0, "test": 0}
    counts_tokens = {"train": 0, "val": 0, "test": 0}

    with open(out_paths["train"], "ab") as f_train, open(out_paths["val"], "ab") as f_val, open(out_paths["test"], "ab") as f_test:
        fps = {"train": f_train, "val": f_val, "test": f_test}

        progress = tqdm(total=MAX_EXAMPLES, desc="Streaming+Encoding", unit="doc")
        for i, row in enumerate(dataset):
            if i >= MAX_EXAMPLES:
                break

            text = extract_text(row)
            if not text:
                progress.update(1)
                continue

            split = pick_split(i, MAX_EXAMPLES)
            toks = encode_text(text)
            buffers[split].extend(toks)
            counts_examples[split] += 1

            # Flush all splits together so val/test are written even if their
            # individual buffers never reach FLUSH_TOKENS (they're only 1% each).
            if len(buffers["train"]) >= FLUSH_TOKENS:
                for s in ("train", "val", "test"):
                    counts_tokens[s] += flush_tokens(fps[s], buffers[s])

            progress.update(1)

        progress.close()

        for split in ("train", "val", "test"):
            counts_tokens[split] += flush_tokens(fps[split], buffers[split])

    print("\nDone.")
    for split in ("train", "val", "test"):
        print(f"{split:>5}: {counts_examples[split]:>10,} docs  ->  {counts_tokens[split]:>12,} tokens")
    print(f"Saved files in: {DATA_DIR.resolve()}")
