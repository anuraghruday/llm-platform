#!/usr/bin/env python3
"""Score the baseline eval_dataset.json with the LLM judge and log to MLflow.

Usage:
    python scripts/run_baseline_eval.py
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow
from evals.llm_judge import llm_judge

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EVAL_PATH = Path("evals/eval_dataset.json")
RESULTS_PATH = Path("evals/baseline_results.json")


def main() -> None:
    records = json.loads(EVAL_PATH.read_text())
    logger.info("Scoring %d records with LLM judge", len(records))

    scored = []
    for i, rec in enumerate(records):
        if not rec.get("answer"):
            logger.warning("Record %d has no answer, skipping", i)
            continue
        scores = llm_judge(rec["question"], rec.get("context", ""), rec["answer"])
        scored.append({**rec, **scores})
        logger.info(
            "[%d/%d] accuracy=%s completeness=%s hallucination=%s",
            i + 1, len(records),
            scores.get("accuracy"), scores.get("completeness"), scores.get("hallucination"),
        )

    avg_acc  = sum(r["accuracy"] for r in scored) / len(scored)
    avg_comp = sum(r["completeness"] for r in scored) / len(scored)
    hall_rate = sum(1 for r in scored if r["hallucination"]) / len(scored)

    logger.info("Baseline — accuracy: %.2f  completeness: %.2f  hallucination_rate: %.3f",
                avg_acc, avg_comp, hall_rate)

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("llm-platform-evals")
    with mlflow.start_run(run_name="baseline-gemini-2.5-flash"):
        mlflow.log_metrics({
            "accuracy": avg_acc,
            "completeness": avg_comp,
            "hallucination_rate": hall_rate,
        })
        mlflow.log_param("model", "gemini-2.5-flash")
        mlflow.log_param("n_questions", len(scored))

    RESULTS_PATH.write_text(json.dumps(scored, indent=2))
    logger.info("Results written to %s", RESULTS_PATH)


if __name__ == "__main__":
    main()
