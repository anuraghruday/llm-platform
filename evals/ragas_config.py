"""Ragas evaluation runner — faithfulness, answer relevancy, context precision and recall."""

import mlflow
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness


def run_ragas_eval(dataset, run_name: str = "ragas-eval") -> dict:
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    metrics = {
        "faithfulness": result["faithfulness"],
        "answer_relevancy": result["answer_relevancy"],
        "context_precision": result["context_precision"],
        "context_recall": result["context_recall"],
    }

    mlflow.set_experiment("llm-platform-evals")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_metrics(metrics)

    return metrics
