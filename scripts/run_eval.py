#!/usr/bin/env python3
"""Run the full evaluation suite against a model endpoint.

Usage:
    python scripts/run_eval.py --endpoint http://localhost:8000 --test_file data/test.jsonl
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from evals.llm_judge import llm_judge

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def query_endpoint(endpoint: str, question: str) -> str:
    response = httpx.post(
        f"{endpoint}/v1/chat/completions",
        json={
            "model": "default",
            "messages": [{"role": "user", "content": question}],
            "stream": False,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://localhost:8000")
    parser.add_argument("--test_file", required=True)
    args = parser.parse_args()

    records = [json.loads(l) for l in open(args.test_file) if l.strip()]
    logger.info("Evaluating %d examples against %s", len(records), args.endpoint)

    results = []
    for i, rec in enumerate(records):
        answer = query_endpoint(args.endpoint, rec["instruction"])
        scores = llm_judge(rec["instruction"], rec.get("context", ""), answer)
        results.append({**rec, "model_answer": answer, **scores})
        if (i + 1) % 20 == 0:
            logger.info("Progress: %d/%d", i + 1, len(records))

    avg_acc = sum(r["accuracy"] for r in results) / len(results)
    avg_cmp = sum(r["completeness"] for r in results) / len(results)
    hall_rate = sum(r["hallucination"] for r in results) / len(results)

    logger.info("Results — accuracy: %.2f  completeness: %.2f  hallucination_rate: %.3f",
                avg_acc, avg_cmp, hall_rate)

    out = Path(args.test_file).with_suffix(".results.jsonl")
    with open(out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    logger.info("Detailed results written to %s", out)


if __name__ == "__main__":
    main()
