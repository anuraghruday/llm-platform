import logging
import warnings
from typing import List

warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document

from src.config import settings
from src.rag.embeddings import get_embeddings

logger = logging.getLogger(__name__)

BATCH_SIZE = 500


def load_from_url(url: str, max_depth: int = 3) -> List[Document]:
    loader = RecursiveUrlLoader(url=url, max_depth=max_depth)
    return loader.load()


def chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(docs)


def ingest_to_chroma(chunks: List[Document]) -> Chroma:
    chunks = filter_complex_metadata(chunks)
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=settings.chroma_db_path,
        embedding_function=embeddings,
    )
    total = len(chunks)
    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        vectorstore.add_documents(batch)
        logger.info("Embedded %d / %d chunks", min(i + BATCH_SIZE, total), total)
    return vectorstore


def get_vectorstore() -> Chroma:
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=settings.chroma_db_path,
        embedding_function=embeddings,
    )
