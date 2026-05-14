"""Script to train and save the query complexity classifier.

Usage:
    python -m src.router.train_classifier --data data/router_labels.jsonl
"""

import argparse
import json
import logging
from pathlib import Path

from src.router.classifier import ComplexityClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to JSONL with {query, label} records")
    parser.add_argument("--out", default="./models/router_v1.pkl")
    args = parser.parse_args()

    records = [json.loads(l) for l in open(args.data) if l.strip()]
    texts = [r["query"] for r in records]
    labels = [r["label"] for r in records]

    logger.info("Training on %d examples (%d routine, %d complex).",
                len(labels), labels.count("routine"), labels.count("complex"))

    clf = ComplexityClassifier()
    clf.fit(texts, labels)
    clf.save(args.out)
    logger.info("Saved classifier to %s", args.out)


if __name__ == "__main__":
    main()
