"""Tests for eval utilities — LLM judge output parsing and scoring."""

import json
from unittest.mock import MagicMock, patch


@patch("evals.llm_judge._get_client")
def test_llm_judge_parses_valid_json(mock_client):
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text='{"accuracy": 4, "completeness": 5, "hallucination": false}')]
    mock_client.return_value.messages.create.return_value = mock_msg

    from evals.llm_judge import llm_judge
    result = llm_judge("What is LoRA?", "LoRA reduces trainable params.", "LoRA is a fine-tuning technique.")
    assert result["accuracy"] == 4
    assert result["completeness"] == 5
    assert result["hallucination"] is False


@patch("evals.llm_judge._get_client")
def test_llm_judge_strips_markdown_fences(mock_client):
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text='```json\n{"accuracy": 3, "completeness": 3, "hallucination": true}\n```')]
    mock_client.return_value.messages.create.return_value = mock_msg

    from evals.llm_judge import llm_judge
    result = llm_judge("Q", "C", "A")
    assert result["hallucination"] is True
