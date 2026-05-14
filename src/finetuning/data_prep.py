"""Format Q&A pairs into Mistral instruction-tuning format for SFTTrainer."""

import json
from pathlib import Path
from typing import Iterator

MISTRAL_TEMPLATE = "<s>[INST] {instruction}\n\nContext:\n{context} [/INST] {response}</s>"


def format_example(instruction: str, context: str, response: str) -> str:
    return MISTRAL_TEMPLATE.format(
        instruction=instruction, context=context, response=response
    )


def load_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def prepare_dataset(raw_path: str | Path, out_path: str | Path) -> int:
    examples = []
    for item in load_jsonl(raw_path):
        text = format_example(
            instruction=item["instruction"],
            context=item.get("context", ""),
            response=item["response"],
        )
        examples.append({"text": text})

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    return len(examples)
