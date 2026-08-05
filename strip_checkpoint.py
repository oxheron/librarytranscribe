#!/usr/bin/env python3
"""Strip a training checkpoint down to what inference needs.

Training checkpoints carry optimizer/scheduler state and training bookkeeping;
transcribe_library.py only reads ckpt["model"] and ckpt["config"]["model"].
Dropping the rest roughly halves the release download.

Usage:
  python strip_checkpoint.py <drumtranscribe>/runs/full_current/best.pt model.pt
"""

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(
        description="Strip a training checkpoint down to what inference needs.")
    parser.add_argument("checkpoint", help="Training checkpoint (best.pt)")
    parser.add_argument("out", help="Slim inference checkpoint to write")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    slim = {"model": ckpt["model"], "config": ckpt.get("config", {})}
    torch.save(slim, args.out)

    before = Path(args.checkpoint).stat().st_size / 2**20
    after = Path(args.out).stat().st_size / 2**20
    print(f"{args.checkpoint} ({before:.0f} MB) -> {args.out} ({after:.0f} MB)")


if __name__ == "__main__":
    main()
