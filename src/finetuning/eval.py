"""Evaluation suite for the fine-tuned model.

Runs Ragas metrics + LLM-as-judge against the held-out test set.
"""

import json
from pathlib import Path

import mlflow
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from evals.llm_judge import llm_judge


def run_ragas(eval_dataset) -> dict:
    result = evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    return {
        "faithfulness": result["faithfulness"],
        "answer_relevancy": result["answer_relevancy"],
        "context_precision": result["context_precision"],
        "context_recall": result["context_recall"],
    }


def run_llm_judge(test_path: str | Path) -> dict:
    records = [json.loads(l) for l in open(test_path) if l.strip()]
    scores = [llm_judge(r["question"], r["context"], r["answer"]) for r in records]
    n = len(scores)
    return {
        "avg_accuracy": sum(s["accuracy"] for s in scores) / n,
        "avg_completeness": sum(s["completeness"] for s in scores) / n,
        "hallucination_rate": sum(s["hallucination"] for s in scores) / n,
    }


def evaluate_model(run_name: str, ragas_dataset, test_path: str | Path) -> dict:
    mlflow.set_experiment("llm-platform-evals")
    with mlflow.start_run(run_name=run_name):
        ragas_metrics = run_ragas(ragas_dataset)
        judge_metrics = run_llm_judge(test_path)
        all_metrics = {**ragas_metrics, **judge_metrics}
        mlflow.log_metrics(all_metrics)
    return all_metrics
