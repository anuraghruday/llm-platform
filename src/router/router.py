"""Adaptive router: sends routine queries to the fine-tuned model, complex ones to Gemini."""

import logging
from pathlib import Path

import httpx

from src.config import settings
from src.router.classifier import ComplexityClassifier, Prediction

logger = logging.getLogger(__name__)

_classifier: ComplexityClassifier | None = None
_CLASSIFIER_PATH = Path("./models/router_v1.pkl")


def _get_classifier() -> ComplexityClassifier:
    global _classifier
    if _classifier is None:
        if _CLASSIFIER_PATH.exists():
            _classifier = ComplexityClassifier.load(_CLASSIFIER_PATH)
        else:
            logger.warning("Router classifier not found at %s — defaulting to Gemini.", _CLASSIFIER_PATH)
            _classifier = None
    return _classifier


def route_query(query: str) -> str:
    """Return 'vllm' or 'gemini' for the given query."""
    classifier = _get_classifier()
    if classifier is None:
        return "gemini"

    pred: Prediction = classifier.predict(query)
    logger.debug("Router: label=%s confidence=%.3f", pred.label, pred.confidence)

    if pred.label == "routine" and pred.confidence >= settings.router_confidence_threshold:
        return "vllm"
    return "gemini"


async def call_vllm(messages: list[dict], max_tokens: int = 1024) -> str:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{settings.vllm_url}/v1/chat/completions",
            json={
                "model": settings.vllm_model,
                "messages": messages,
                "max_tokens": max_tokens,
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
