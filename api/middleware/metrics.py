"""
Prometheus metrics for observability.

WHY THIS EXISTS:
Without metrics, you're flying blind in production. You can't answer:
- How many requests per second are we handling?
- What's the P99 latency?
- What's the error rate?
- How many tokens are we generating per second?

This module exposes Prometheus-compatible metrics at /metrics that can
be scraped by Prometheus and visualized in Grafana.

Metrics exposed:
- http_requests_total: counter of requests by method/path/status
- http_request_duration_seconds: histogram of request latency
- http_requests_in_progress: gauge of concurrent requests
- model_generation_total: counter of generations by model/strategy
- model_generation_duration_seconds: histogram of generation time
"""

import time

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# ── Metrics Registry ────────────────────────────────────────────────────
REGISTRY = CollectorRegistry()

# Counter: always increases (total requests, total errors)
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
    registry=REGISTRY,
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=REGISTRY,
)

http_requests_in_progress = Gauge(
    "http_requests_in_progress",
    "Number of HTTP requests currently in progress",
    registry=REGISTRY,
)

model_generation_total = Counter(
    "model_generation_total",
    "Total model generations",
    ["model", "strategy"],
    registry=REGISTRY,
)

model_generation_duration_seconds = Histogram(
    "model_generation_duration_seconds",
    "Model generation duration in seconds",
    ["model"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0),
    registry=REGISTRY,
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Collect HTTP metrics for every request."""

    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip metrics endpoint itself to avoid infinite loop
        if request.url.path == "/metrics":
            return await call_next(request)

        http_requests_in_progress.inc()
        start_time = time.time()

        try:
            response = await call_next(request)
            status = response.status_code
        except Exception:
            status = 500
            raise
        finally:
            duration = time.time() - start_time
            http_requests_in_progress.dec()

            # Normalize path to avoid high cardinality
            # /api/v1/models/123 → /api/v1/models/:id
            path = request.url.path
            # Simple normalization: replace numeric segments with :id
            segments = path.split("/")
            normalized = [":id" if seg.isdigit() else seg for seg in segments]
            path = "/".join(normalized)

            http_requests_total.labels(
                method=request.method,
                path=path,
                status=str(status),
            ).inc()

            http_request_duration_seconds.labels(
                method=request.method,
                path=path,
            ).observe(duration)

        return response


def record_generation(model: str, strategy: str, duration: float) -> None:
    """Record a model generation for metrics.

    Call this from the chat route after each generation completes.
    """
    model_generation_total.labels(model=model, strategy=strategy).inc()
    model_generation_duration_seconds.labels(model=model).observe(duration)


def get_metrics() -> bytes:
    """Return Prometheus-formatted metrics for /metrics endpoint."""
    return generate_latest(REGISTRY)
