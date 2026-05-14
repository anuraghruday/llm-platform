"""Alerting thresholds and helpers.

Prometheus alerting rules live in k8s/monitoring/prometheus-config.yaml.
This module provides Python-side alert definitions for reference.
"""

ALERT_RULES = [
    {
        "name": "HighErrorRate",
        "expr": 'rate(llm_requests_total{status="error"}[5m]) / rate(llm_requests_total[5m]) > 0.05',
        "for": "2m",
        "severity": "critical",
        "summary": "Error rate > 5% over 5 min",
    },
    {
        "name": "HighP95Latency",
        "expr": 'histogram_quantile(0.95, rate(llm_request_latency_seconds_bucket[5m])) > 5',
        "for": "5m",
        "severity": "warning",
        "summary": "p95 latency > 5s",
    },
    {
        "name": "GuardrailSpike",
        "expr": "rate(llm_guardrail_triggers_total[5m]) > 1",
        "for": "1m",
        "severity": "warning",
        "summary": "Guardrail firing > 1/s",
    },
]
