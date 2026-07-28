#!/usr/bin/env python3
"""
Prepare instruction-tuning dataset for chat fine-tuning.
Uses tatsu-lab/alpaca (52K instruction examples) and formats with
System:/User:/Assistant: markers.
"""
import os
from pathlib import Path
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm.auto import tqdm

DATA_DIR = Path("instruction_data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token


def format_instruction(row):
    """Format a single instruction example."""
    instruction = row["instruction"].strip()
    input_text = row.get("input", "").strip()
    output = row["output"].strip()

    if input_text:
        text = f"System: You are a helpful assistant.\nUser: {instruction}\n{input_text}\nAssistant: {output}"
    else:
        text = f"System: You are a helpful assistant.\nUser: {instruction}\nAssistant: {output}"

    return text


print("Loading Alpaca dataset...")
dataset = load_dataset("tatsu-lab/alpaca")

# Alpaca has only train split — create our own val/test split
train_data = dataset["train"]

# Split 95/2.5/2.5
n = len(train_data)
n_test = n // 40   # 2.5%
n_val = n // 40    # 2.5%
n_train = n - n_val - n_test

splits = {
    "train": train_data.select(range(n_train)),
    "val": train_data.select(range(n_train, n_train + n_val)),
    "test": train_data.select(range(n_train + n_val, n)),
}

out_paths = {
    "train": DATA_DIR / "train.bin",
    "val": DATA_DIR / "val.bin",
    "test": DATA_DIR / "test.bin",
}

for split_name, split_data in splits.items():
    print(f"Processing {split_name} split ({len(split_data)} examples)...")
    tokens = []

    for row in tqdm(split_data, desc=f"Encoding {split_name}"):
        text = format_instruction(row)
        ids = enc.encode_ordinary(text)
        ids.append(EOT)
        tokens.extend(ids)

    arr = np.asarray(tokens, dtype=np.uint16)
    arr.tofile(out_paths[split_name])
    print(f"  Saved {len(tokens):,} tokens to {out_paths[split_name]}")

print("\nInstruction data preparation complete.")
print(f"Files saved to: {DATA_DIR}")
