"""Prometheus metrics — counters, histograms, and gauges for the API."""

from prometheus_client import Counter, Gauge, Histogram

# Request metrics
REQUEST_COUNT = Counter(
    "llm_requests_total", "Total requests", ["model", "status"]
)
REQUEST_LATENCY = Histogram(
    "llm_request_latency_seconds", "End-to-end request latency", ["model"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)
TTFT = Histogram(
    "llm_ttft_seconds", "Time to first token", ["model"],
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
)

# Cost metrics
TOKEN_COST = Counter("llm_token_cost_usd_total", "Cumulative token cost (USD)", ["model"])
TOKENS_USED = Counter("llm_tokens_used_total", "Total tokens consumed", ["model", "type"])

# Quality / safety
GUARDRAIL_TRIGGERS = Counter(
    "llm_guardrail_triggers_total", "Guardrail trigger count", ["category"]
)

# Routing
ROUTER_DECISIONS = Counter(
    "llm_router_decisions_total", "Routing decisions", ["target"]
)
ROUTER_CONFIDENCE = Histogram(
    "llm_router_confidence", "Classifier confidence at routing time",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0],
)

# Semantic cache (future)
CACHE_HITS = Counter("llm_cache_hits_total", "Semantic cache hits")
CACHE_MISSES = Counter("llm_cache_misses_total", "Semantic cache misses")

# vLLM throughput gauge
VLLM_THROUGHPUT = Gauge("vllm_tokens_per_second", "vLLM generation throughput")
