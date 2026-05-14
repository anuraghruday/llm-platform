"""Tests for the adaptive router."""

from unittest.mock import MagicMock, patch
from src.router.classifier import Prediction


@patch("src.router.router._get_classifier")
def test_routes_routine_high_confidence_to_vllm(mock_get):
    mock_clf = MagicMock()
    mock_clf.predict.return_value = Prediction(label="routine", confidence=0.95)
    mock_get.return_value = mock_clf

    from src.router.router import route_query
    assert route_query("How do I install FastAPI?") == "vllm"


@patch("src.router.router._get_classifier")
def test_routes_complex_to_gemini(mock_get):
    mock_clf = MagicMock()
    mock_clf.predict.return_value = Prediction(label="complex", confidence=0.92)
    mock_get.return_value = mock_clf

    from src.router.router import route_query
    assert route_query("Explain the architectural tradeoffs of LCEL vs LangGraph.") == "gemini"


@patch("src.router.router._get_classifier")
def test_routes_low_confidence_routine_to_gemini(mock_get):
    mock_clf = MagicMock()
    mock_clf.predict.return_value = Prediction(label="routine", confidence=0.65)
    mock_get.return_value = mock_clf

    from src.router.router import route_query
    assert route_query("What does asyncio.gather do?") == "gemini"


@patch("src.router.router._get_classifier")
def test_no_classifier_defaults_to_gemini(mock_get):
    mock_get.return_value = None

    from src.router.router import route_query
    assert route_query("Any query") == "gemini"
