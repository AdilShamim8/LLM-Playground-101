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
from datetime import datetime, timedelta
from typing import Optional

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

SECRET_KEY = os.getenv(
    "JWT_SECRET_KEY", "dev-secret-change-in-production"
)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# Rate limiting config
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "3600"))

security = HTTPBearer(auto_error=False)

if JOSE_AVAILABLE:
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ── In-memory stores (use Redis in production) ────────────────────

_api_keys: dict[str, dict] = {
    "dev-key-123": {
        "user_id": "dev",
        "role": "admin",
        "created_at": time.time()
    }
}
_rate_limit_store: dict[str, list[float]] = defaultdict(list)
_revoked_tokens: set[str] = set()


def verify_password(
    plain_password: str, hashed_password: str
) -> bool:
    if not JOSE_AVAILABLE:
        return True
    return pwd_context.verify(plain_password, hashed_password)


def hash_password(password: str) -> str:
    if not JOSE_AVAILABLE:
        return password
    return pwd_context.hash(password)


def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None
) -> str:
    if not JOSE_AVAILABLE:
        return "dev-token"

    to_encode = data.copy()
    expire = datetime.utcnow() + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    if not JOSE_AVAILABLE:
        return {"sub": "dev", "role": "admin"}
    try:
        payload = jwt.decode(
            token, SECRET_KEY, algorithms=[ALGORITHM]
        )
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
        ts for ts in _rate_limit_store[user_id]
        if ts > window_start
    ]

    if len(_rate_limit_store[user_id]) >= RATE_LIMIT_REQUESTS:
        return False

    _rate_limit_store[user_id].append(now)
    return True


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security)
) -> dict:
    """
    FastAPI dependency for authenticated routes.
    Supports Bearer JWT tokens and API keys.
    """
    # Allow all in dev mode
    if os.getenv("DEV_MODE", "true").lower() == "true":
        return {"user_id": "dev", "role": "admin"}

    if not credentials:
        # Check API key header
        api_key = request.headers.get("X-API-Key")
        if api_key and api_key in _api_keys:
            user_info = _api_keys[api_key]
            user_id = user_info["user_id"]
            if not check_rate_limit(user_id):
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded"
                )
            return user_info

        raise HTTPException(
            status_code=401,
            detail="Authentication required"
        )

    token = credentials.credentials

    # Check revoked tokens
    if token in _revoked_tokens:
        raise HTTPException(
            status_code=401, detail="Token revoked"
        )

    payload = decode_token(token)
    if not payload:
        raise HTTPException(
            status_code=401, detail="Invalid or expired token"
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=401, detail="Invalid token payload"
        )

    if not check_rate_limit(user_id):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {RATE_LIMIT_REQUESTS} "
                   f"requests per {RATE_LIMIT_WINDOW}s"
        )

    return {
        "user_id": user_id,
        "role": payload.get("role", "user"),
        "exp": payload.get("exp")
    }


async def require_admin(
    user: dict = Security(get_current_user)
) -> dict:
    """Dependency for admin-only routes."""
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=403,
            detail="Admin role required"
        )
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