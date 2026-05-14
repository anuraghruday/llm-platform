from langchain_core.tools import tool

from src.rag.retriever import format_context, retrieve


@tool
def search_docs(query: str) -> str:
    """Search the documentation corpus for passages relevant to the query."""
    docs = retrieve(query)
    if not docs:
        return "No relevant documentation found."
    return format_context(docs)


@tool
def calculator(expression: str) -> str:
    """Evaluate a safe mathematical expression (e.g. '2 ** 10 + 42')."""
    allowed = set("0123456789+-*/()., ")
    if not all(c in allowed for c in expression):
        return "Error: only basic arithmetic is supported (+, -, *, /, parentheses)."
    try:
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return str(result)
    except ZeroDivisionError:
        return "Error: division by zero."
    except Exception as exc:
        return f"Error: {exc}"


TOOLS = [search_docs, calculator]
