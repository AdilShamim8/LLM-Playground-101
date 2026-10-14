"""
Centralized configuration management using Pydantic Settings.

WHY THIS EXISTS:
Previously, environment variables were read with os.getenv() scattered
across multiple files (auth.py, main.py, routes/*.py). This made it
impossible to know what configuration the app needed, and there was no
validation at startup.

With Pydantic Settings:
- All config is defined in ONE place (this file)
- Type validation at startup (fail fast if env var is wrong type)
- Default values are explicit
- Settings can be overridden via .env file or environment variables
- IDE autocomplete works on config fields

Usage:
    from config.settings import settings
    if settings.DEV_MODE:
        ...
    db_url = settings.DATABASE_URL
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ── Core ────────────────────────────────────────────────────────────
    PROJECT_NAME: str = "llm_playground"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = False
    DEV_MODE: bool = False

    # ── API ─────────────────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 1

    # ── Authentication ──────────────────────────────────────────────────
    JWT_SECRET_KEY: str = Field(
        default="",
        description="Generate with: python -c 'import secrets; print(secrets.token_urlsafe(32))'",
    )
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 1440  # 24 hours

    # ── Rate Limiting ────────────────────────────────────────────────────
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 3600

    # ── Database ────────────────────────────────────────────────────────
    POSTGRES_USER: str = "llm_user"
    POSTGRES_PASSWORD: str = ""
    POSTGRES_DB: str = "llm_playground"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432

    # ── Redis ───────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── Model Storage ───────────────────────────────────────────────────
    MODEL_CHECKPOINT_DIR: str = "./checkpoints"
    DATA_DIR: str = "./data"
    TOKENIZER_DIR: str = "./tokenizers"

    # ── CORS ────────────────────────────────────────────────────────────
    CORS_ORIGINS: str = ""  # Comma-separated list

    # ── Training ────────────────────────────────────────────────────────
    CUDA_VISIBLE_DEVICES: str = "0"
    WANDB_PROJECT: str = "llm_playground"
    WANDB_API_KEY: str = ""

    # ── UI ──────────────────────────────────────────────────────────────
    UI_PORT: int = 7860
    API_BASE: str = "http://localhost:8000/api/v1"

    # ── Logging ─────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/app.log"

    # ── Computed fields ─────────────────────────────────────────────────

    @computed_field
    @property
    def DATABASE_URL(self) -> str:
        """Construct the full database URL from components."""
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @computed_field
    @property
    def CORS_ORIGINS_LIST(self) -> list[str]:
        """Parse CORS_ORIGINS string into a list."""
        if not self.CORS_ORIGINS:
            return []
        return [
            origin.strip()
            for origin in self.CORS_ORIGINS.split(",")
            if origin.strip()
        ]

    # ── Validation ──────────────────────────────────────────────────────

    def validate_production(self) -> None:
        """Fail-fast validation for production environment.

        Call this at app startup. If any required setting is missing or
        invalid for production, raise RuntimeError with a helpful message.
        """
        if self.ENVIRONMENT != "production":
            return

        errors = []

        if self.DEV_MODE:
            errors.append(
                "DEV_MODE=true in production. Set DEV_MODE=false."
            )

        if not self.JWT_SECRET_KEY or len(self.JWT_SECRET_KEY) < 32:
            errors.append(
                "JWT_SECRET_KEY must be set and at least 32 characters. "
                "Generate with: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
            )

        if not self.POSTGRES_PASSWORD:
            errors.append("POSTGRES_PASSWORD must be set in production.")

        if not self.CORS_ORIGINS:
            errors.append(
                "CORS_ORIGINS must be set in production. "
                "Example: CORS_ORIGINS=https://app.example.com"
            )

        if errors:
            raise RuntimeError(
                "Production configuration validation failed:\n  - "
                + "\n  - ".join(errors)
            )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    lru_cache ensures we only parse env vars once per process.
    Call settings() again to get the same instance.
    """
    return Settings()


# Global settings instance (imported everywhere)
settings = get_settings()
