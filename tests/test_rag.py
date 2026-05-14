"""Tests for the RAG pipeline."""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from src.rag.retriever import format_context


def test_format_context_includes_source():
    docs = [
        Document(page_content="FastAPI uses Pydantic for validation.", metadata={"source": "https://fastapi.tiangolo.com/"}),
        Document(page_content="Starlette is the underlying ASGI framework.", metadata={"source": "https://fastapi.tiangolo.com/advanced/"}),
    ]
    result = format_context(docs)
    assert "FastAPI uses Pydantic" in result
    assert "fastapi.tiangolo.com" in result
    assert "Starlette" in result


def test_format_context_missing_source():
    docs = [Document(page_content="Some content.", metadata={})]
    result = format_context(docs)
    assert "unknown" in result
    assert "Some content." in result


@patch("src.rag.retriever._get_vectorstore")
def test_retrieve_calls_vectorstore(mock_vs):
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        Document(page_content="LangChain is a framework.", metadata={"source": "lc"})
    ]
    mock_vs.return_value.as_retriever.return_value = mock_retriever

    from src.rag.retriever import retrieve
    results = retrieve("What is LangChain?")
    assert len(results) == 1
    assert "LangChain" in results[0].page_content
