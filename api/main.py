"""
FastAPI backend for LLM Playground.
Production-ready REST API with:
- Authentication (JWT)
- Rate limiting
- Async chat streaming
- Model management
- Evaluation endpoints
"""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from loguru import logger

from api.routes.chat import router as chat_router
from api.routes.models import router as models_router
from api.routes.evaluation import router as eval_router
from api.middleware.auth import auth_middleware


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api/v1", tags=["chat"])
app.include_router(
    models_router, prefix="/api/v1", tags=["models"]
)
app.include_router(eval_router, prefix="/api/v1", tags=["eval"])


@app.get("/")
async def root():
    return {
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
        "api_base": "/api/v1"
    }


@app.get("/health")
async def health_check(request: Request):
    uptime = time.time() - request.app.state.start_time
    return {
        "status": "healthy",
        "uptime": uptime,
        "requests": request.app.state.request_count,
        "version": "1.0.0"
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
            "path": str(request.url)
        }
    )