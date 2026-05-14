#!/usr/bin/env python3
"""Generate Q&A training pairs from ChromaDB document chunks.

For each chunk, uses Gemini to generate 3 question-answer pairs and
writes them to a JSONL file formatted for SFTTrainer.

Usage:
    python scripts/generate_training_data.py --out data/raw_qa.jsonl --limit 3000
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google import genai
from google.genai import types

from src.config import settings
from src.rag.ingest import get_vectorstore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """\
Given the documentation excerpt below, generate exactly 3 question-answer pairs that test \
understanding of the content. Return ONLY a JSON array, no commentary.

Each element: {{"question": "...", "answer": "..."}}

Documentation:
{chunk}"""


def generate_qa(client: genai.Client, chunk: str) -> list[dict]:
    response = client.models.generate_content(
        model=settings.gemini_model_fast,
        contents=PROMPT_TEMPLATE.format(chunk=chunk),
        config=types.GenerateContentConfig(max_output_tokens=1024),
    )
    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("```")[1].lstrip("json").strip()
    pairs = json.loads(text)
    return [{"instruction": p["question"], "context": chunk, "response": p["answer"]} for p in pairs]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw_qa.jsonl")
    parser.add_argument("--limit", type=int, default=3000, help="Max chunks to process")
    args = parser.parse_args()

    client = genai.Client(api_key=settings.google_api_key)
    vs = get_vectorstore()
    collection = vs._collection
    results = collection.get(limit=args.limit)
    chunks = results["documents"] or []

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    total = 0

    with open(args.out, "w") as f:
        for i, chunk in enumerate(chunks):
            try:
                pairs = generate_qa(client, chunk)
                for pair in pairs:
                    f.write(json.dumps(pair) + "\n")
                total += len(pairs)
                if (i + 1) % 50 == 0:
                    logger.info("Processed %d/%d chunks — %d pairs so far", i + 1, len(chunks), total)
            except Exception as exc:
                logger.warning("Chunk %d failed: %s", i, exc)

    logger.info("Done. %d Q&A pairs written to %s", total, args.out)


if __name__ == "__main__":
    main()
