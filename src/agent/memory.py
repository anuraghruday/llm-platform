"""In-memory conversation session store, keyed by session_id."""

from collections import defaultdict
from typing import Dict, List

from langchain_core.messages import BaseMessage

_store: Dict[str, List[BaseMessage]] = defaultdict(list)

MAX_HISTORY = 30


def get_history(session_id: str) -> List[BaseMessage]:
    return list(_store[session_id])


def save_messages(session_id: str, messages: List[BaseMessage]) -> None:
    _store[session_id].extend(messages)
    if len(_store[session_id]) > MAX_HISTORY:
        _store[session_id] = _store[session_id][-MAX_HISTORY:]


def clear_session(session_id: str) -> None:
    _store.pop(session_id, None)


def session_count() -> int:
    return len(_store)
