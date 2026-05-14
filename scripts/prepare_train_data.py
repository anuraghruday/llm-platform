#!/usr/bin/env python3
"""Format split JSONL files into SFTTrainer-ready format.

Reads data/{split}.jsonl (instruction/context/response) and writes
data/{split}_formatted.jsonl (text field using Mistral chat template).

Usage:
    python scripts/prepare_train_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.finetuning.data_prep import prepare_dataset

DATA_DIR = Path("data")

for split in ("train", "val", "test"):
    src = DATA_DIR / f"{split}.jsonl"
    dst = DATA_DIR / f"{split}_formatted.jsonl"
    if src.exists():
        n = prepare_dataset(src, dst)
        print(f"{split}: {n} examples → {dst}")
    else:
        print(f"{split}: {src} not found, skipping")
