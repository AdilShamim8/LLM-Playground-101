"""
Security headers middleware.

Adds HTTP security headers to every response to protect against common
web vulnerabilities:
- X-Content-Type-Options: prevents MIME-type sniffing
- X-Frame-Options: prevents clickjacking
- X-XSS-Protection: enables browser XSS filtering
- Referrer-Policy: controls referrer information
- Strict-Transport-Security: forces HTTPS (only in production)
- Content-Security-Policy: restricts resource loading (optional)
"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)

        # Prevent MIME-type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"

        # Prevent clickjacking (page can't be embedded in iframe)
        response.headers["X-Frame-Options"] = "SAMEORIGIN"

        # Enable browser XSS filtering
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Control referrer information sent to other sites
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Force HTTPS in production (1 year, include subdomains)
        from config.settings import settings
        if settings.ENVIRONMENT == "production":
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains; preload"
            )

        return response
