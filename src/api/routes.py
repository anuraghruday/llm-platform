import json
import time
import uuid
from typing import AsyncGenerator, List, Literal, Optional

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from pydantic import BaseModel

from src.agent.graph import agent
from src.agent.guardrails import FALLBACK_RESPONSE, SafetyResult, check_safety
from src.agent.memory import get_history, save_messages
from src.config import Settings, get_settings

router = APIRouter()


# ── Pydantic models ────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    model: str = "gemini-2.5-flash"
    messages: List[Message]
    temperature: Optional[float] = 1.0
    stream: Optional[bool] = True
    max_tokens: Optional[int] = 1024
    session_id: Optional[str] = None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _last_user_content(messages: List[Message]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    return ""


def _final_ai_text(messages: List[AnyMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
            content = m.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(p if isinstance(p, str) else p.get("text", "") for p in content)
    return ""


def _sse_chunk(text: str, model: str) -> str:
    payload = {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
    }
    return f"data: {json.dumps(payload)}\n\n"


def _sse_done(model: str, session_id: str, guardrail: SafetyResult) -> str:
    payload = {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "session_id": session_id,
        "guardrail": {"safe": guardrail.safe, "category": guardrail.category},
    }
    return f"data: {json.dumps(payload)}\n\ndata: [DONE]\n\n"


# ── Streaming agent runner ─────────────────────────────────────────────────────

async def _agent_stream(
    session_id: str,
    history: List[AnyMessage],
    user_msg: HumanMessage,
    query_safety: SafetyResult,
    request: ChatRequest,
) -> AsyncGenerator[str, None]:
    if not query_safety.safe:
        yield _sse_chunk(FALLBACK_RESPONSE, request.model)
        yield _sse_done(request.model, session_id, query_safety)
        return

    final_text = ""
    async for event in agent.astream_events(
        {"messages": history + [user_msg]},
        version="v2",
    ):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            text = chunk.content if isinstance(chunk.content, str) else ""
            if text and not getattr(chunk, "tool_call_chunks", None):
                final_text += text
                yield _sse_chunk(text, request.model)

    save_messages(session_id, [user_msg, AIMessage(content=final_text)])
    yield _sse_done(request.model, session_id, SafetyResult(safe=True, category="none"))


# ── Endpoint ───────────────────────────────────────────────────────────────────

@router.post("/v1/chat/completions")
async def chat_completions(
    request: ChatRequest,
    settings: Settings = Depends(get_settings),
):
    session_id = request.session_id or str(uuid.uuid4())
    history = get_history(session_id)
    user_content = _last_user_content(request.messages)
    user_msg = HumanMessage(content=user_content)

    # Pre-check: query safety (applies to both streaming and non-streaming)
    query_safety = await check_safety(user_content)

    if request.stream:
        return StreamingResponse(
            _agent_stream(session_id, history, user_msg, query_safety, request),
            media_type="text/event-stream",
        )

    # Non-streaming ────────────────────────────────────────────────────────────
    if not query_safety.safe:
        return _non_stream_response(FALLBACK_RESPONSE, request, session_id, query_safety)

    result = await agent.ainvoke({"messages": history + [user_msg]})
    response_text = _final_ai_text(result["messages"])

    # Post-check: response safety
    response_safety = await check_safety("", response_text)
    if not response_safety.safe:
        response_text = FALLBACK_RESPONSE

    final_safety = response_safety if not query_safety.safe else response_safety
    new_msgs = result["messages"][len(history):]
    save_messages(session_id, new_msgs)

    return _non_stream_response(response_text, request, session_id, final_safety)


def _non_stream_response(text: str, request: ChatRequest, session_id: str, safety: SafetyResult) -> dict:
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "session_id": session_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "guardrail": {"safe": safety.safe, "category": safety.category},
    }
