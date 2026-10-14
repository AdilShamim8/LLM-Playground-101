"""
Structured logging configuration using loguru.

WHY THIS EXISTS:
The app was using loguru's default configuration, which outputs to stderr
with a simple format. In production, you need:
- Structured JSON logs for log aggregation (ELK, Datadog, CloudWatch)
- Log rotation to prevent disk filling
- Correlation IDs for tracing requests across services
- Different log levels per environment

Usage:
    from config.logging import setup_logging
    setup_logging()

    from loguru import logger
    logger.info("Model loaded", model="gpt-7b", params="7B")
"""

import sys
from pathlib import Path

from loguru import logger

from config.settings import settings


def setup_logging() -> None:
    """Configure loguru logging with structured output and rotation.

    Call this once at application startup (in api/main.py lifespan).
    """
    # Remove default handler
    logger.remove()

    # ── Console handler (human-readable) ───────────────────────────────
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stderr,
        format=console_format,
        level=settings.LOG_LEVEL,
        colorize=True,
        backtrace=True,
        diagnose=settings.DEBUG,  # Show variable values in traceback (dev only)
    )

    # ── File handler with rotation ─────────────────────────────────────
    log_path = Path(settings.LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        str(log_path),
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        ),
        level=settings.LOG_LEVEL,
        rotation="100 MB",       # Rotate when file reaches 100MB
        retention="30 days",     # Keep logs for 30 days
        compression="zip",       # Compress rotated logs
        backtrace=True,
        diagnose=False,          # Never show variable values in file logs (security)
    )

    # ── JSON handler for production log aggregation ────────────────────
    if settings.ENVIRONMENT == "production":
        json_log_path = log_path.parent / "app.jsonl"
        logger.add(
            str(json_log_path),
            serialize=True,       # JSON format for ELK/Datadog/CloudWatch
            level="INFO",        # Only INFO+ in JSON logs
            rotation="100 MB",
            retention="30 days",
            compression="zip",
        )

    logger.info(
        "Logging initialized",
        environment=settings.ENVIRONMENT,
        log_level=settings.LOG_LEVEL,
        log_file=str(log_path),
    )
