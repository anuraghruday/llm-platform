import logging
from typing import List

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from src.config import settings
from src.rag.ingest import get_vectorstore

logger = logging.getLogger(__name__)

_vectorstore: Chroma | None = None


def _get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = get_vectorstore()
    return _vectorstore


def retrieve(query: str) -> List[Document]:
    docs = _get_vectorstore().as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.retrieval_top_k},
    ).invoke(query)
    logger.debug("Retrieved %d docs for query: %.60s", len(docs), query)
    return docs


def format_context(docs: List[Document]) -> str:
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )
