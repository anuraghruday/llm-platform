#!/usr/bin/env python3
"""Cost and latency benchmark.

Sends N queries to the platform endpoint, measures latency, cost, and
routing decisions, then prints a summary table.

Usage:
    python scripts/benchmark.py --endpoint http://localhost:8000 --n 500
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

SAMPLE_QUERIES = [
    "How do I define a Pydantic model with optional fields?",
    "Explain LangChain's LCEL pipe syntax.",
    "What is the difference between QLoRA and standard LoRA?",
    "How do I add a middleware to a FastAPI app?",
    "What does HuggingFace's pipeline() function do?",
    "How can I stream tokens from a Gemini API call in Python?",
    "What is ChromaDB and how does it store embeddings?",
    "Explain Python asyncio event loop in simple terms.",
    "How do I fine-tune a model with the Trainer API?",
    "What is RAG and why is it useful?",
]


def run_benchmark(endpoint: str, n: int) -> None:
    latencies = []
    errors = 0

    for i in range(n):
        query = SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)]
        start = time.perf_counter()
        try:
            r = httpx.post(
                f"{endpoint}/v1/chat/completions",
                json={"model": "default", "messages": [{"role": "user", "content": query}], "stream": False},
                timeout=60,
            )
            r.raise_for_status()
            latencies.append(time.perf_counter() - start)
        except Exception as exc:
            errors += 1
            print(f"[{i}] Error: {exc}")

        if (i + 1) % 50 == 0:
            print(f"Progress: {i + 1}/{n}")

    if not latencies:
        print("No successful requests.")
        return

    latencies.sort()
    p = lambda q: latencies[int(len(latencies) * q)]
    print(f"\n{'='*50}")
    print(f"Queries: {n}  |  Errors: {errors}")
    print(f"p50 latency : {p(0.50)*1000:.0f} ms")
    print(f"p95 latency : {p(0.95)*1000:.0f} ms")
    print(f"p99 latency : {p(0.99)*1000:.0f} ms")
    print(f"Mean latency: {statistics.mean(latencies)*1000:.0f} ms")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()
    run_benchmark(args.endpoint, args.n)
