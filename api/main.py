"""
FastAPI backend for LLM Playground.

Production-ready REST API with:
- Authentication (JWT + API keys)
- Rate limiting (Redis-backed in production)
- Security headers
- Request ID tracing
- Prometheus metrics
- Structured logging with rotation
- Global exception handling
- Async chat streaming (non-blocking)
- Model management
- Evaluation endpoints
- Health checks (liveness + readiness)
"""

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from loguru import logger

from api.middleware.exceptions import global_exception_handler
from api.middleware.metrics import PrometheusMiddleware, get_metrics
from api.middleware.request_id import RequestIDMiddleware
from api.middleware.security_headers import SecurityHeadersMiddleware
from api.routes.chat import router as chat_router
from api.routes.evaluation import router as eval_router
from api.routes.health import router as health_router, set_app_ready
from api.routes.models import router as models_router

# ── Application version ─────────────────────────────────────────────────
__version__ = "2.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application startup and shutdown lifecycle.

    On startup:
    - Initialize structured logging
    - Validate production configuration
    - Initialize database (if configured)
    - Signal readiness

    On shutdown:
    - Log shutdown event
    """
    # ── Startup ──────────────────────────────────────────────────────────
    from config.logging import setup_logging

    setup_logging()

    logger.info(f"Starting LLM Playground API v{__version__}...")

    # Validate production configuration (fail-fast if misconfigured)
    from config.settings import settings

    try:
        settings.validate_production()
        logger.info(
            f"Configuration validated | Environment: {settings.ENVIRONMENT}"
        )
    except RuntimeError as e:
        logger.error(f"Configuration validation failed:\n{e}")
        raise

    app.state.start_time = time.time()
    app.state.request_count = 0

    # Signal readiness (for /ready endpoint)
    set_app_ready(True)

    logger.info("LLM Playground API is ready to serve traffic")

    yield

    # ── Shutdown ────────────────────────────────────────────────────────
    set_app_ready(False)
    logger.info("Shutting down LLM Playground API...")


# ── Create FastAPI app ───────────────────────────────────────────────────
app = FastAPI(
    title="LLM Playground API",
    description=(
        "Production-level LLM Playground with "
        "pre-training, SFT, RLHF, and evaluation."
    ),
    version=__version__,
    lifespan=lifespan,
)

# ── Middleware stack (order matters — outermost first) ───────────────────
# Request ID must be first so all subsequent middleware/logs have it
app.add_middleware(RequestIDMiddleware)

# Security headers on every response
app.add_middleware(SecurityHeadersMiddleware)

# Prometheus metrics collection
app.add_middleware(PrometheusMiddleware)

# CORS (must be after security middleware so headers aren't stripped)
from config.settings import settings

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS_LIST or [
        "http://localhost:7860",
        "http://localhost:3000",
        "http://127.0.0.1:7860",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)

# ── Routes ───────────────────────────────────────────────────────────────
app.include_router(chat_router, prefix="/api/v1", tags=["chat"])
app.include_router(models_router, prefix="/api/v1", tags=["models"])
app.include_router(eval_router, prefix="/api/v1", tags=["eval"])
app.include_router(health_router, tags=["health"])


# ── Root endpoint ────────────────────────────────────────────────────────
@app.get("/")
async def root() -> dict:
    """Root endpoint with API info and available endpoints."""
    return {
        "status": "ok",
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
        "ready": "/ready",
        "metrics": "/metrics",
        "api_base": "/api/v1",
    }


# ── Metrics endpoint (Prometheus scraping) ───────────────────────────────
@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus-format metrics endpoint.

    Scraped by Prometheus every 15s. Returns counters, histograms,
    and gauges for HTTP requests and model generations.
    """
    return Response(
        content=get_metrics(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


# ── Request counter middleware ──────────────────────────────────────────
@app.middleware("http")
async def count_requests(request: Request, call_next):
    """Count total requests for the /health endpoint stats."""
    request.app.state.request_count += 1
    response = await call_next(request)
    return response


# ── Exception handlers ──────────────────────────────────────────────────
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle expected HTTP exceptions with clean JSON response."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url.path),
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Catch all unhandled exceptions, log them, return clean error.

    This prevents raw tracebacks from leaking to users in production.
    """
    return await global_exception_handler(request, exc)
