"""
Global exception handler middleware.

WHY THIS EXISTS:
Without a global exception handler, unhandled errors return a 500 with
a raw traceback to the user. This:
- Leaks internal details (security risk)
- Provides poor UX (raw Python error message)
- Doesn't log the error for debugging

This middleware catches ALL unhandled exceptions, logs them with context,
and returns a clean JSON error response.
"""

import uuid

from fastapi import Request
from fastapi.responses import JSONResponse
from loguru import logger


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch all unhandled exceptions and return a clean JSON error.

    Logs the full exception with stack trace for debugging, but only
    returns a generic error message to the user (no internal details).
    """
    error_id = str(uuid.uuid4())

    logger.error(
        f"Unhandled exception | ID={error_id} | "
        f"Path={request.url.path} | Method={request.method} | "
        f"Error={type(exc).__name__}: {exc}",
        exc_info=True,
    )

    # In development, include the error message for debugging
    # In production, return only a generic message + error ID for correlation
    from config.settings import settings

    if settings.DEBUG:
        detail = str(exc)
    else:
        detail = f"An internal error occurred. Reference ID: {error_id}"

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": detail,
            "error_id": error_id,
            "path": str(request.url.path),
        },
    )
