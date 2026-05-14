# LLM Platform

A self-hosted LLM API with RAG, tool-use, safety filtering, and a fine-tuning + eval pipeline. Exposes an OpenAI-compatible `/v1/chat/completions` endpoint so any existing client works without modification.

---

## What it does

- **Answers questions grounded in real documentation** — scrapes Python, LangChain, FastAPI, and HuggingFace docs, chunks and embeds them locally, stores in ChromaDB. Every response is backed by retrieved context, not just model memory.
- **Reasons in steps, not one shot** — a LangGraph agent decides when to search docs, when to do math, and when it has enough to answer. It loops until confident rather than guessing on the first pass.
- **Remembers conversations** — sessions are keyed by `session_id`, so follow-up questions work naturally.
- **Filters unsafe content** — Gemini classifies both the incoming query and outgoing response. Prompt injection, harmful requests, and explicit content are caught and blocked before they reach the user.
- **Streams responses** — SSE streaming out of the box, same format as OpenAI.
- **Tracks everything** — MLflow logs every fine-tuning run, eval score, and model version. Promotion to production is gated on eval thresholds, not manual review.

---

## Architecture

```
Client
  │
  ▼
FastAPI  (/v1/chat/completions)
  │
  ├── Safety pre-check (Gemini Flash classifier)
  │
  ▼
LangGraph Agent
  ├── search_docs → ChromaDB (all-MiniLM-L6-v2 embeddings, 1913 chunks)
  └── calculator  → sandboxed eval()
  │
  ├── Safety post-check
  │
  ▼
SSE stream → Client
```

The fine-tuning pipeline runs separately on a cloud GPU. Training data is generated from the same ChromaDB corpus using Gemini, formatted into Mistral's instruction template, and trained with QLoRA. Eval runs before and after using an LLM-as-judge setup, with results logged to MLflow for comparison.

---

## Tech stack

| Layer | What |
|-------|------|
| API | FastAPI, Uvicorn, SSE streaming |
| Agent | LangGraph, LangChain tools |
| LLM | Gemini 2.5 Flash (via google-genai SDK) |
| Retrieval | ChromaDB, all-MiniLM-L6-v2 (HuggingFace, local CPU) |
| Fine-tuning | Mistral-7B-Instruct-v0.3, QLoRA (bitsandbytes + peft), SFTTrainer |
| Eval | LLM-as-judge (Gemini Flash), Ragas metrics |
| Experiment tracking | MLflow |
| Config | pydantic-settings |

---

## Getting started

Python 3.11 required. The API runs on CPU for development — no GPU needed locally. Fine-tuning needs a CUDA GPU (Colab notebook included).

```bash
conda create -n llm-platform python=3.11
conda activate llm-platform
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your Google API key (free tier at [aistudio.google.com](https://aistudio.google.com) works fine):

```
GOOGLE_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash
GEMINI_MODEL_FAST=gemini-2.5-flash
```

Build the doc index (one-time, ~2 minutes):

```bash
python scripts/ingest_docs.py
```

Start the server:

```bash
uvicorn src.api.main:app --reload --port 8000
```

---

## Usage

Health check:
```bash
curl http://localhost:8000/health
```

Single question (non-streaming):
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "How does LoRA reduce trainable parameters?"}],
    "stream": false
  }'
```

Multi-turn conversation — pass the same `session_id` across requests:
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is ChromaDB?"}],
    "session_id": "my-session-123",
    "stream": false
  }'
```

Streaming works the same way, just set `"stream": true` and read the SSE events.

---

## Project layout

```
src/
  config.py              # pydantic-settings, lru_cache singleton
  api/
    main.py              # FastAPI app, startup, CORS, middleware
    routes.py            # /v1/chat/completions, streaming + non-streaming
    middleware.py        # request logging (X-Request-Id), Prometheus stub
  rag/
    embeddings.py        # HuggingFace embedder singleton (CPU)
    ingest.py            # load URLs → chunk → embed → ChromaDB
    retriever.py         # similarity search, context formatter
  agent/
    graph.py             # LangGraph state machine
    tools.py             # search_docs, calculator
    guardrails.py        # Gemini safety classifier, pre + post check
    memory.py            # in-memory session store
    prompts.py           # system prompt
  finetuning/
    data_prep.py         # Mistral [INST] template formatter
    train.py             # QLoRA training script (Linux/GPU)
    promote.py           # MLflow model promotion logic

scripts/
  ingest_docs.py              # scrape and index documentation
  generate_training_data.py   # generate Q&A pairs from the corpus via Gemini
  split_data.py               # 80/10/10 train/val/test split
  prepare_train_data.py       # format splits for SFTTrainer
  build_eval_dataset.py       # collect baseline answers from the live API
  run_baseline_eval.py        # score with LLM judge, log to MLflow
  promote_model.py            # CLI for MLflow model promotion

evals/
  llm_judge.py          # Gemini-as-judge: accuracy, completeness, hallucination
  ragas_config.py       # Ragas metrics (faithfulness, answer relevancy, etc.)

notebooks/
  finetune_colab.ipynb  # end-to-end QLoRA training on Google Colab
```

---

## Running tests

```bash
pytest
```

Covers RAG retrieval, agent tool execution, safety classifier, and session memory.

---

## Fine-tuning

The fine-tuning code targets Linux + CUDA. Use the included Colab notebook if you don't have a local GPU.

**Prepare the data locally:**
```bash
python scripts/generate_training_data.py --limit 200
python scripts/split_data.py --input data/raw_qa.jsonl
python scripts/prepare_train_data.py
```

**Train on Colab:**
Upload `data/train_formatted.jsonl` and `data/val_formatted.jsonl`, open `notebooks/finetune_colab.ipynb`, set your HuggingFace token (required for the Mistral gated model), and run all cells. The checkpoint downloads at the end.

**Evaluate and promote:**
```bash
python scripts/run_baseline_eval.py
python scripts/promote_model.py --run_id <mlflow_run_id>
```

**Baseline eval (pre fine-tuning, Gemini 2.5 Flash):**

| Metric | Score |
|--------|-------|
| Accuracy | 4.20 / 5 |
| Completeness | 3.87 / 5 |
| Hallucination rate | 6.7% |

The same 15 questions are re-scored after fine-tuning. If the numbers hold or improve, the model is promoted to production in the MLflow registry.

---

## A few things worth noting

The ChromaDB index and training data aren't committed — they're generated artifacts that add up to ~100MB of binary files. Run the ingest and data generation scripts to rebuild them.

`requirements_train.txt` has the GPU training dependencies separately. Don't install those on your dev machine — they pull in large CUDA libraries that aren't needed for running the API.

`gemini-2.5-pro` requires a GCP project with billing enabled. The `.env.example` defaults to `gemini-2.5-flash` which works on the free tier.
