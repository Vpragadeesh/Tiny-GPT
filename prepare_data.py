#!/usr/bin/env python3
"""
prepare_data.py
===============
Build tokenized binary files from Cosmopedia using streaming.

Outputs:
  data/train.bin
  data/val.bin
  data/test.bin

Dataset:
  HuggingFaceTB/cosmopedia

This script streams records and writes token ids directly to .bin files,
so RAM usage stays low even with very large datasets.
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
DATASET_NAME = "HuggingFaceTB/cosmopedia"
DATASET_SPLIT = os.environ.get("DATASET_SPLIT", "train[:1000000]")
TRAIN_FRAC = float(os.environ.get("TRAIN_FRAC", "0.98"))
VAL_FRAC = float(os.environ.get("VAL_FRAC", "0.01"))

enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token


def encode_text(text: str):
    ids = enc.encode_ordinary(text)
    ids.append(EOT)
    return np.asarray(ids, dtype=np.uint16)


def split_name(i: int, total: int):
    train_cut = int(total * TRAIN_FRAC)
    val_cut = train_cut + int(total * VAL_FRAC)
    if i < train_cut:
        return "train"
    if i < val_cut:
        return "val"
    #!/usr/bin/env python3
    """
    prepare_data.py
    ===============
    Build tokenized binary files for training from Cosmopedia using streaming.

    Outputs:
      data/train.bin
      data/val.bin
      data/test.bin

    Dataset:
      HuggingFaceTB/cosmopedia (streaming)
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
    DATASET_NAME = "HuggingFaceTB/cosmopedia"

    # Stream only first N rows by default, matching user request intent.
    MAX_EXAMPLES = int(os.environ.get("MAX_EXAMPLES", "1000000"))

    # Deterministic split from one stream: 98% train, 1% val, 1% test.
    VAL_EVERY = 100
    TEST_EVERY = 100
    VAL_SLOT = 0
    TEST_SLOT = 1

    FLUSH_TOKENS = int(os.environ.get("FLUSH_TOKENS", "2000000"))

    enc = tiktoken.get_encoding("gpt2")
    EOT = enc.eot_token


    def extract_text(row: dict) -> str:
        """Extract a usable text field across possible Cosmopedia schemas."""
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


    if __name__ == "__main__":
        print("Loading Cosmopedia (streaming)...")
        dataset = load_dataset(DATASET_NAME, split="train", streaming=True, cache_dir=CACHE_DIR)

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
                if MAX_EXAMPLES > 0 and i >= MAX_EXAMPLES:
                    break

                text = extract_text(row)
                if not text:
                    progress.update(1)
                    continue

                if i % VAL_EVERY == VAL_SLOT:
                    split = "val"
                elif i % TEST_EVERY == TEST_SLOT:
                    split = "test"
                else:
                    split = "train"

                toks = encode_text(text)
                buffers[split].extend(toks)
                counts_examples[split] += 1

                if len(buffers[split]) >= FLUSH_TOKENS:
                    counts_tokens[split] += flush_tokens(fps[split], buffers[split])

                progress.update(1)

            progress.close()

            for split in ("train", "val", "test"):
                counts_tokens[split] += flush_tokens(fps[split], buffers[split])

        print("\nDone.")
        for split in ("train", "val", "test"):
            print(
                f"{split:>5}: {counts_examples[split]:>10,} docs  ->  {counts_tokens[split]:>12,} tokens"
            )
        print(f"Saved files in: {DATA_DIR.resolve()}")
