#!/usr/bin/env python3
"""One-time document ingestion: scrapes docs and loads them into ChromaDB."""

import logging
import sys
import warnings
from pathlib import Path

from bs4 import XMLParsedAsHTMLWarning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.ingest import chunk_documents, ingest_to_chroma, load_from_url

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SOURCES = [
    ("Python docs",         "https://docs.python.org/3/",                    1),
    ("LangChain docs",      "https://python.langchain.com/docs/",            1),
    ("FastAPI docs",        "https://fastapi.tiangolo.com/",                 1),
    ("HuggingFace docs",    "https://huggingface.co/docs/transformers/",     1),
]


def main() -> None:
    all_chunks = []

    for name, url, depth in SOURCES:
        logger.info("Loading %s from %s (depth=%d)...", name, url, depth)
        try:
            docs = load_from_url(url, max_depth=depth)
            chunks = chunk_documents(docs)
            logger.info("  %s: %d pages -> %d chunks", name, len(docs), len(chunks))
            all_chunks.extend(chunks)
        except Exception as exc:
            logger.error("  Failed to load %s: %s", name, exc)

    if not all_chunks:
        logger.error("No chunks produced — aborting.")
        sys.exit(1)

    logger.info("Total chunks: %d. Ingesting into ChromaDB...", len(all_chunks))
    ingest_to_chroma(all_chunks)
    logger.info("Done. ChromaDB populated at configured path.")


if __name__ == "__main__":
    main()
