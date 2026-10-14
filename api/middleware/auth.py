"""
JWT Authentication middleware for LLM Playground API.
Production-ready with:
- JWT token generation and validation
- API key support
- Rate limiting per user
- Role-based access control
"""

import os
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger

try:
    from jose import JWTError, jwt
    from passlib.context import CryptContext

    JOSE_AVAILABLE = True
except ImportError:
    JOSE_AVAILABLE = False
    logger.warning("jose/passlib not installed. Auth disabled.")

# ── Environment-based configuration ────────────────────────────────────
#
# SECURITY: We do NOT provide a fallback default for JWT_SECRET_KEY.
# If the env var is missing, the app should fail to start in production.
# A hardcoded default secret is how APIs get hacked — anyone who reads
# the public repo knows the secret and can forge JWT tokens.
#
# The only exception is DEV_MODE, where a known dev secret is acceptable
# for local development. In production, DEV_MODE must be false.

DEV_MODE = os.getenv("DEV_MODE", "false").lower() == "true"
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

if DEV_MODE:
    # In dev mode only, use a non-secret placeholder.
    # This NEVER runs in production because we check ENVIRONMENT below.
    SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-only-not-for-production")
    if ENVIRONMENT == "production":
        raise RuntimeError(
            "FATAL: DEV_MODE=true in production. "
            "Set DEV_MODE=false and provide JWT_SECRET_KEY."
        )
else:
    SECRET_KEY = os.getenv("JWT_SECRET_KEY")
    if not SECRET_KEY:
        raise RuntimeError(
            "FATAL: JWT_SECRET_KEY environment variable is not set. "
            'Generate one with: python -c "import secrets; print(secrets.token_urlsafe(32))"'
        )
    if len(SECRET_KEY) < 32:
        raise RuntimeError(
            "FATAL: JWT_SECRET_KEY must be at least 32 characters. "
            "Current length: " + str(len(SECRET_KEY))
        )

ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(
    os.getenv("JWT_EXPIRE_MINUTES", "1440")  # 24 hours
)

# Rate limiting config (per-user, per-window)
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "3600"))

security = HTTPBearer(auto_error=False)

if JOSE_AVAILABLE:
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ── In-memory stores ───────────────────────────────────────────────────
#
# WARNING: These are process-local. In production with multiple workers
# or multiple servers, use Redis instead. See docs/PRODUCTION_DEPLOYMENT.md.
#
# API keys should be loaded from a database or vault, NOT hardcoded.
# In dev mode, we allow a dev key for convenience. In production, the
# _api_keys dict starts empty and must be populated via an admin API.

_api_keys: dict[str, dict] = {}
if DEV_MODE:
    _api_keys["dev-key-123"] = {
        "user_id": "dev",
        "role": "admin",
        "created_at": time.time(),
    }
    logger.warning(
        "DEV_MODE is ON — a hardcoded dev API key is active. "
        "NEVER use this in production."
    )

_rate_limit_store: dict[str, list[float]] = defaultdict(list)
_revoked_tokens: set[str] = set()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    if not JOSE_AVAILABLE:
        return True
    return pwd_context.verify(plain_password, hashed_password)


def hash_password(password: str) -> str:
    if not JOSE_AVAILABLE:
        return password
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    if not JOSE_AVAILABLE:
        return "dev-token"

    to_encode = data.copy()
    # datetime.utcnow() is deprecated in Python 3.12+.
    # Use timezone-aware datetime instead.
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict | None:
    if not JOSE_AVAILABLE:
        return {"sub": "dev", "role": "admin"}
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


def check_rate_limit(user_id: str) -> bool:
    """
    Sliding window rate limiter.
    Returns True if within limit, False if exceeded.
    """
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW

    # Clean old entries
    _rate_limit_store[user_id] = [
        ts for ts in _rate_limit_store[user_id] if ts > window_start
    ]

    if len(_rate_limit_store[user_id]) >= RATE_LIMIT_REQUESTS:
        return False

    _rate_limit_store[user_id].append(now)
    return True


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Security(security),
) -> dict:
    """
    FastAPI dependency for authenticated routes.
    Supports Bearer JWT tokens and API keys.
    """
    # In dev mode, skip authentication for convenience.
    # The DEV_MODE flag is checked at import time (see above) and will
    # raise RuntimeError if ENVIRONMENT=production.
    if DEV_MODE:
        return {"user_id": "dev", "role": "admin"}

    if not credentials:
        # Check API key header
        api_key = request.headers.get("X-API-Key")
        if api_key and api_key in _api_keys:
            user_info = _api_keys[api_key]
            user_id = user_info["user_id"]
            if not check_rate_limit(user_id):
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            return user_info

        raise HTTPException(status_code=401, detail="Authentication required")

    token = credentials.credentials

    # Check revoked tokens
    if token in _revoked_tokens:
        raise HTTPException(status_code=401, detail="Token revoked")

    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    if not check_rate_limit(user_id):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {RATE_LIMIT_REQUESTS} "
            f"requests per {RATE_LIMIT_WINDOW}s",
        )

    return {
        "user_id": user_id,
        "role": payload.get("role", "user"),
        "exp": payload.get("exp"),
    }


async def require_admin(user: dict = Security(get_current_user)) -> dict:
    """Dependency for admin-only routes."""
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    return user


async def auth_middleware(request: Request, call_next):
    """
    Global middleware to log all requests.
    Actual auth is handled per-route via dependencies.
    """
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start

    logger.info(
        f"{request.method} {request.url.path} "
        f"-> {response.status_code} "
        f"({duration*1000:.1f}ms)"
    )
    return response
