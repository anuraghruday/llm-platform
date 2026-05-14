#!/usr/bin/env python3
"""Split raw Q&A JSONL into train / val / test sets (80 / 10 / 10).

Usage:
    python scripts/split_data.py --input data/raw_qa.jsonl
Outputs:
    data/train.jsonl  data/val.jsonl  data/test.jsonl
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw_qa.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = [json.loads(l) for l in open(args.input) if l.strip()]
    random.seed(args.seed)
    random.shuffle(records)

    n = len(records)
    n_val  = max(1, int(n * 0.10))
    n_test = max(1, int(n * 0.10))
    n_train = n - n_val - n_test

    splits = {
        "train": records[:n_train],
        "val":   records[n_train : n_train + n_val],
        "test":  records[n_train + n_val :],
    }

    out_dir = Path(args.input).parent
    for name, rows in splits.items():
        out = out_dir / f"{name}.jsonl"
        with open(out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{name:6s}: {len(rows):>5d} examples → {out}")

    print(f"\nTotal: {n} examples split {n_train}/{n_val}/{n_test}")


if __name__ == "__main__":
    main()
