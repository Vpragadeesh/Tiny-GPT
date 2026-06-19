#!/usr/bin/env python3
"""
Upload Tiny-GPT checkpoints to Hugging Face Hub.

Usage:
  python push_to_hf.py --repo-id yourname/Tiny-GPT
  python push_to_hf.py --repo-id yourname/Tiny-GPT --checkpoint checkpoints/best.pt

Auth:
  Set HF_TOKEN env var or run: huggingface-cli login
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Upload Tiny-GPT checkpoints to HF Hub")
    parser.add_argument("--repo-id", required=True, help="HF repo id, e.g. yourname/Tiny-GPT")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best.pt",
        help="Primary checkpoint path to upload (default: checkpoints/best.pt)",
    )
    parser.add_argument(
        "--latest-checkpoint",
        default="checkpoints/latest.pt",
        help="Optional latest checkpoint path to upload (default: checkpoints/latest.pt)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private repo instead of public",
    )
    parser.add_argument(
        "--message",
        default="Upload Tiny-GPT checkpoints",
        help="Commit message for HF Hub",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF token (or set HF_TOKEN env var)",
    )
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")

    try:
        from huggingface_hub import HfApi, upload_file
    except ImportError:
        print("[ERROR] Missing dependency: huggingface_hub")
        print("[ERROR] Install with: pip install huggingface_hub")
        sys.exit(1)

    checkpoint = Path(args.checkpoint)
    latest_checkpoint = Path(args.latest_checkpoint)

    if not checkpoint.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint}")
        sys.exit(1)

    api = HfApi(token=token)

    # Create repo if it does not exist yet.
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    print(f"Uploading {checkpoint} -> {args.repo_id}/best.pt")
    upload_file(
        path_or_fileobj=str(checkpoint),
        path_in_repo="best.pt",
        repo_id=args.repo_id,
        repo_type="model",
        token=token,
        commit_message=args.message,
    )

    if latest_checkpoint.exists():
        print(f"Uploading {latest_checkpoint} -> {args.repo_id}/latest.pt")
        upload_file(
            path_or_fileobj=str(latest_checkpoint),
            path_in_repo="latest.pt",
            repo_id=args.repo_id,
            repo_type="model",
            token=token,
            commit_message=args.message,
        )

    print("Done. Model checkpoints are now on Hugging Face Hub.")


if __name__ == "__main__":
    main()
