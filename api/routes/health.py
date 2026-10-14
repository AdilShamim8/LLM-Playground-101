"""
Enhanced health check endpoints.

WHY THIS EXISTS:
A single /health endpoint is insufficient for production. Kubernetes and
other orchestrators distinguish between:
- Liveness: "Is the process alive?" (if fails, restart the container)
- Readiness: "Is the app ready to serve traffic?" (if fails, remove from
  load balancer but don't restart)

An app might be alive (process running) but NOT ready (database not
connected, model not loaded). These endpoints handle that distinction.
"""

import time

from fastapi import APIRouter
from loguru import logger

router = APIRouter()

# Track app start time for uptime calculation
_START_TIME = time.time()

# Track app readiness state (set by app during startup)
_app_ready: bool = False


def set_app_ready(ready: bool) -> None:
    """Called by the app to signal readiness state changes."""
    global _app_ready
    _app_ready = ready
    logger.info(f"App readiness: {'READY' if ready else 'NOT READY'}")


@router.get("/health")
async def health() -> dict:
    """Basic liveness check.

    Returns 200 if the process is alive and can respond to HTTP.
    Used by Kubernetes livenessProbe — if this fails, the container
    is restarted.
    """
    return {
        "status": "healthy",
        "uptime_seconds": time.time() - _START_TIME,
    }


@router.get("/ready")
async def readiness() -> dict:
    """Readiness check.

    Returns 200 if the app is ready to serve traffic (model loaded,
    database connected, etc.). Used by Kubernetes readinessProbe —
    if this fails, the container is removed from the load balancer
    but NOT restarted.
    """
    if not _app_ready:
        return {
            "status": "not_ready",
            "reason": "Model not loaded or dependencies not connected",
        }
    return {
        "status": "ready",
        "uptime_seconds": time.time() - _START_TIME,
    }


@router.get("/health/detailed")
async def detailed_health() -> dict:
    """Detailed health with component status.

    Checks individual dependencies (database, redis, model).
    Useful for debugging — shows which component is failing.
    """
    checks = {}

    # Check Redis (if configured)
    try:
        import redis
        from config.settings import settings
        r = redis.from_url(settings.REDIS_URL, socket_timeout=2)
        r.ping()
        checks["redis"] = "healthy"
    except Exception as e:
        checks["redis"] = f"unhealthy: {e}"

    # Check model registry
    try:
        from api.routes.chat import ModelRegistry
        models = ModelRegistry.list_models()
        checks["models_loaded"] = len(models)
        checks["model_registry"] = "healthy"
    except Exception as e:
        checks["model_registry"] = f"unhealthy: {e}"

    # Overall status
    all_healthy = all(
        v == "healthy" or isinstance(v, int)
        for v in checks.values()
    )

    return {
        "status": "healthy" if all_healthy else "degraded",
        "uptime_seconds": time.time() - _START_TIME,
        "components": checks,
    }
