from typing import List
from langchain_core.documents import Document

def rerank(query: str, docs: List[Document], top_k: int = 3) -> List[Document]:
    return docs[:top_k]
