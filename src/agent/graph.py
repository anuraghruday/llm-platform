"""LangGraph agent — multi-step reasoning with tool use.

Flow: reason → [tools → reason]* → END
"""

from typing import Annotated

from langchain_core.messages import AnyMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from src.agent.prompts import SYSTEM_PROMPT
from src.agent.tools import TOOLS
from src.config import settings


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


llm = ChatGoogleGenerativeAI(
    model=settings.gemini_model,
    google_api_key=settings.google_api_key,
).bind_tools(TOOLS)


def reason_node(state: AgentState) -> dict:
    messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
    response = llm.invoke(messages)
    return {"messages": [response]}


def should_use_tools(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "respond"


tool_node = ToolNode(TOOLS)

workflow = StateGraph(AgentState)
workflow.add_node("reason", reason_node)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("reason")
workflow.add_conditional_edges("reason", should_use_tools, {"tools": "tools", "respond": END})
workflow.add_edge("tools", "reason")

agent = workflow.compile()
