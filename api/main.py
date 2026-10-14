"""
FastAPI backend for LLM Playground.
Production-ready REST API with:
- Authentication (JWT)
- Rate limiting
- Async chat streaming
- Model management
- Evaluation endpoints
"""

import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from api.routes.chat import router as chat_router
from api.routes.evaluation import router as eval_router
from api.routes.models import router as models_router

# ── CORS Configuration ───────────────────────────────────────────────────
#
# SECURITY: allow_origins=["*"] + allow_credentials=True is INVALID per the
# CORS spec (https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS/Errors).
# Browsers will reject credentialed requests from wildcard origins.
#
# In production, set CORS_ORIGINS to a comma-separated list of allowed origins:
#   CORS_ORIGINS=https://app.example.com,https://www.example.com
#
# In development, we default to localhost for the UI and API docs.


def _get_cors_origins() -> list[str]:
    """Parse allowed CORS origins from environment variable."""
    raw = os.getenv("CORS_ORIGINS", "")
    if raw:
        # Explicit list provided — use it
        return [origin.strip() for origin in raw.split(",") if origin.strip()]
    # Default: allow localhost for development only
    return [
        "http://localhost:7860",  # Gradio UI
        "http://localhost:3000",  # Next.js / React dev server
        "http://localhost:8000",  # FastAPI docs
        "http://127.0.0.1:7860",
        "http://127.0.0.1:3000",
    ]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application startup and shutdown lifecycle."""
    logger.info("Starting LLM Playground API...")
    app.state.start_time = time.time()
    app.state.request_count = 0
    yield
    logger.info("Shutting down LLM Playground API...")


app = FastAPI(
    title="LLM Playground API",
    description=(
        "Production-level LLM Playground with "
        "pre-training, SFT, RLHF, and evaluation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS with explicit origins (never wildcard + credentials)
cors_origins = _get_cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)
logger.info(f"CORS allowed origins: {cors_origins}")

app.include_router(chat_router, prefix="/api/v1", tags=["chat"])
app.include_router(models_router, prefix="/api/v1", tags=["models"])
app.include_router(eval_router, prefix="/api/v1", tags=["eval"])


@app.get("/")
async def root():
    return {"status": "ok", "docs": "/docs", "health": "/health", "api_base": "/api/v1"}


@app.get("/health")
async def health_check(request: Request):
    uptime = time.time() - request.app.state.start_time
    return {
        "status": "healthy",
        "uptime": uptime,
        "requests": request.app.state.request_count,
        "version": "1.0.0",
    }


@app.middleware("http")
async def count_requests(request: Request, call_next):
    request.app.state.request_count += 1
    response = await call_next(request)
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url),
        },
    )
