"""MLflow auto-promotion: gates model registry stage transitions on eval thresholds."""

import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "llm-platform-mistral"

THRESHOLDS = {
    "faithfulness": 0.88,
    "answer_relevancy": 0.85,
    "hallucination_rate": 0.01,  # max allowed
}


def maybe_promote(run_id: str) -> bool:
    client = MlflowClient()
    run = client.get_run(run_id)
    m = run.data.metrics

    passes = (
        m.get("faithfulness", 0) >= THRESHOLDS["faithfulness"]
        and m.get("answer_relevancy", 0) >= THRESHOLDS["answer_relevancy"]
        and m.get("hallucination_rate", 1) <= THRESHOLDS["hallucination_rate"]
    )

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    latest = max(int(v.version) for v in versions) if versions else None

    if passes and latest:
        client.transition_model_version_stage(
            name=MODEL_NAME, version=str(latest), stage="Production"
        )
    elif latest:
        client.transition_model_version_stage(
            name=MODEL_NAME, version=str(latest), stage="Staging"
        )

    return passes
