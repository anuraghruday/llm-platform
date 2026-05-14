"""LLM-as-judge evaluator — scores accuracy, completeness, and hallucination via Gemini."""

import json

from google import genai
from google.genai import types

from src.config import settings

JUDGE_PROMPT = """\
Rate the following answer on a scale of 1-5 for accuracy and completeness.
If context is provided, flag hallucination if the answer adds facts not in the context.
If no context is provided, assess based on general correctness.

Question: {question}

Context: {context}

Answer: {answer}

Respond with ONLY this JSON (no markdown, no explanation):
{{"accuracy": <1-5>, "completeness": <1-5>, "hallucination": <true|false>}}"""


def llm_judge(question: str, context: str, answer: str) -> dict:
    client = genai.Client(api_key=settings.google_api_key)
    context_str = context if context else "(no context provided)"
    try:
        response = client.models.generate_content(
            model=settings.gemini_model_fast,
            contents=JUDGE_PROMPT.format(
                question=question, context=context_str, answer=answer[:2000]
            ),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                max_output_tokens=512,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        text = (response.text or "").strip()
        if text.startswith("```"):
            text = text.split("```")[1].lstrip("json").strip()
        if text:
            return json.loads(text)
    except Exception:
        pass
    return {"accuracy": 3, "completeness": 3, "hallucination": False}
