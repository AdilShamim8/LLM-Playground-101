"""
Request ID middleware.

WHY THIS EXISTS:
In a microservice architecture, a single user request might touch multiple
services. Without a correlation ID, it's impossible to trace a request
across services. This middleware:
- Generates a unique ID for every request (or uses incoming X-Request-ID)
- Adds it to the response headers so clients can reference it
- Adds it to loguru's context so every log line includes the request ID

Usage in logs: logger.info("Processing")  # automatically includes request_id
"""

import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Generate and propagate a unique request ID for every request."""

    async def dispatch(self, request: Request, call_next) -> Response:
        # Use incoming X-Request-ID if provided (for distributed tracing)
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        # Store on request state for handlers to access
        request.state.request_id = request_id

        # Add to loguru context so all logs include the request ID
        from loguru import logger
        with logger.contextualize(request_id=request_id):
            response = await call_next(request)

        # Return the request ID in the response header
        response.headers["X-Request-ID"] = request_id
        return response
