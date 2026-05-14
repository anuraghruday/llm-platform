#!/usr/bin/env python3
"""Populate eval_dataset.json with real answers from the running API.

Run this BEFORE fine-tuning to establish a baseline.
The same questions will be re-scored after fine-tuning for comparison.

Usage:
    python scripts/build_eval_dataset.py --endpoint http://localhost:8000
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

EVAL_PATH = Path("evals/eval_dataset.json")

QUESTIONS = [
    "How do I create a streaming FastAPI endpoint?",
    "What is the difference between LangChain's RunnableParallel and RunnableSequence?",
    "How do I load a HuggingFace model with 4-bit quantization using bitsandbytes?",
    "What are Python's asyncio.gather() and asyncio.wait() and when should I prefer one?",
    "How does LoRA reduce the number of trainable parameters compared to full fine-tuning?",
    "How do I add middleware to a FastAPI application?",
    "What is ChromaDB and how does it store embeddings?",
    "How do I use LangChain's RecursiveCharacterTextSplitter?",
    "What is the difference between Pydantic v1 and v2 model validation?",
    "How do I implement a custom LangChain tool?",
    "What is vLLM and why is it faster than naive HuggingFace generation?",
    "How do I use the HuggingFace pipeline() function for text generation?",
    "What is RAG and what problem does it solve?",
    "How do I handle CORS in FastAPI?",
    "What is the purpose of the @lru_cache decorator in Python?",
]


def query(endpoint: str, question: str) -> tuple[str, str]:
    r = httpx.post(
        f"{endpoint}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": question}], "stream": False},
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    answer = data["choices"][0]["message"]["content"]
    return answer, ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://localhost:8000")
    args = parser.parse_args()

    records = []
    for i, q in enumerate(QUESTIONS, 1):
        print(f"[{i}/{len(QUESTIONS)}] {q[:60]}...")
        try:
            answer, context = query(args.endpoint, q)
            records.append({"question": q, "context": context, "answer": answer})
        except Exception as exc:
            print(f"  FAILED: {exc}")
            records.append({"question": q, "context": "", "answer": ""})

    EVAL_PATH.write_text(json.dumps(records, indent=2))
    print(f"\nWrote {len(records)} records to {EVAL_PATH}")


if __name__ == "__main__":
    main()
