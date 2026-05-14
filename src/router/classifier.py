"""Query complexity classifier.

Trains a logistic regression on sentence embeddings to label queries
as 'routine' or 'complex'. Serialises to a pickle file for fast loading.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.rag.embeddings import get_embeddings


@dataclass
class Prediction:
    label: str       # "routine" | "complex"
    confidence: float


class ComplexityClassifier:
    def __init__(self):
        self._pipe: Pipeline | None = None

    def fit(self, texts: list[str], labels: list[str]) -> None:
        embeddings = get_embeddings()
        X = np.array(embeddings.embed_documents(texts))
        self._pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, C=1.0)),
        ])
        self._pipe.fit(X, labels)

    def predict(self, text: str) -> Prediction:
        if self._pipe is None:
            raise RuntimeError("Classifier not trained — call fit() or load() first.")
        embeddings = get_embeddings()
        X = np.array(embeddings.embed_query(text)).reshape(1, -1)
        label = self._pipe.predict(X)[0]
        proba = self._pipe.predict_proba(X)[0]
        confidence = float(max(proba))
        return Prediction(label=label, confidence=confidence)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._pipe, f)

    @classmethod
    def load(cls, path: str | Path) -> "ComplexityClassifier":
        instance = cls()
        with open(path, "rb") as f:
            instance._pipe = pickle.load(f)  # noqa: S301
        return instance
