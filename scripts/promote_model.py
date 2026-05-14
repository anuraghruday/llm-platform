#!/usr/bin/env python3
"""Check eval metrics for a fine-tuning run and promote/stage in MLflow.

Usage:
    python scripts/promote_model.py --run_id <mlflow_run_id>
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.finetuning.promote import maybe_promote


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True, help="MLflow run ID from training")
    args = parser.parse_args()

    promoted = maybe_promote(args.run_id)
    if promoted:
        print("Model PROMOTED to Production — all thresholds passed.")
    else:
        print("Model moved to Staging — one or more thresholds not met.")


if __name__ == "__main__":
    main()
