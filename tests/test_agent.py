"""Tests for agent components — tools, guardrails, and session memory."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agent.memory import clear_session, get_history, save_messages, session_count
from src.agent.tools import calculator


# ── Memory tests ───────────────────────────────────────────────────────────────

def test_memory_stores_and_retrieves():
    save_messages("sess-1", [HumanMessage(content="Hello")])
    history = get_history("sess-1")
    assert len(history) == 1
    assert history[0].content == "Hello"
    clear_session("sess-1")


def test_memory_clear_session():
    save_messages("sess-del", [HumanMessage(content="temp")])
    clear_session("sess-del")
    assert get_history("sess-del") == []


def test_memory_independent_sessions():
    save_messages("a", [HumanMessage(content="A")])
    save_messages("b", [HumanMessage(content="B")])
    assert get_history("a")[0].content == "A"
    assert get_history("b")[0].content == "B"
    clear_session("a")
    clear_session("b")


# ── Tool tests ─────────────────────────────────────────────────────────────────

def test_calculator_basic():
    assert calculator.invoke({"expression": "2 ** 10"}) == "1024"


def test_calculator_float():
    result = calculator.invoke({"expression": "1 / 3"})
    assert "0.333" in result


def test_calculator_blocks_unsafe():
    result = calculator.invoke({"expression": "__import__('os').getcwd()"})
    assert "Error" in result


# ── Guardrail tests ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_guardrail_safe_response():
    with patch("src.agent.guardrails.genai.Client") as mock_client_cls:
        mock_result = MagicMock()
        mock_result.text = '{"safe": true, "category": "none", "reason": "ok"}'
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_result)
        mock_client_cls.return_value = mock_client

        from src.agent.guardrails import check_safety
        result = await check_safety("How do I use FastAPI?", "You can use FastAPI like this...")
        assert result.safe is True
        assert result.category == "none"


@pytest.mark.asyncio
async def test_guardrail_unsafe_response():
    with patch("src.agent.guardrails.genai.Client") as mock_client_cls:
        mock_result = MagicMock()
        mock_result.text = '{"safe": false, "category": "harmful", "reason": "dangerous"}'
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_result)
        mock_client_cls.return_value = mock_client

        from src.agent.guardrails import check_safety
        result = await check_safety("How do I harm someone?", "Here is how...")
        assert result.safe is False
        assert result.category == "harmful"


@pytest.mark.asyncio
async def test_guardrail_defaults_safe_on_api_error():
    with patch("src.agent.guardrails.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(side_effect=Exception("timeout"))
        mock_client_cls.return_value = mock_client

        from src.agent.guardrails import check_safety
        result = await check_safety("What is Python?", "Python is a language.")
        assert result.safe is True
