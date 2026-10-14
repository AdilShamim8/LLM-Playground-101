"""
Shared model registry — single source of truth for loaded models.

WHY THREAD SAFETY MATTERS:
In production with multiple uvicorn workers or concurrent requests,
multiple threads might try to load/unload models simultaneously. Without
locking:
- Thread A loads model "gpt7b" → 5 seconds
- Thread B loads model "gpt7b" simultaneously → duplicate work, GPU OOM
- Thread C unloads model "gpt7b" while Thread A is using it → crash

This module uses threading.RLock to make all operations thread-safe.

NOTE: This is process-local. For multi-worker or multi-node setups, use
a Redis-backed registry or a model server like vLLM with Ray Serve.
"""

import threading
import time
from typing import Optional


class ModelRegistry:
    """
    Thread-safe in-memory model registry.

    Stores loaded models, tokenizers, and generators with metadata
    (path, device, dtype, load time). All operations are protected by
    a re-entrant lock for thread safety.

    For production multi-worker setups, replace with a Redis-backed
    registry or use vLLM with Ray Serve for model orchestration.
    """

    _models: dict[str, dict] = {}
    _lock: threading.RLock = threading.RLock()

    @classmethod
    def register(
        cls,
        name: str,
        model,
        tokenizer,
        generator,
        path: str = "",
        device: str = "unknown",
        dtype: str = "unknown",
    ) -> None:
        """Register or replace a loaded model.

        Thread-safe: if two threads register the same name simultaneously,
        the last one wins (no corruption).
        """
        with cls._lock:
            cls._models[name] = {
                "model": model,
                "tokenizer": tokenizer,
                "generator": generator,
                "path": path,
                "device": device,
                "dtype": dtype,
                "loaded_at": time.time(),
            }

    @classmethod
    def get(cls, name: str) -> Optional[dict]:
        """Get a registered model's info. Returns None if not found."""
        with cls._lock:
            return cls._models.get(name)

    @classmethod
    def list_models(cls) -> list[str]:
        """List all registered model names."""
        with cls._lock:
            return list(cls._models.keys())

    @classmethod
    def unregister(cls, name: str) -> bool:
        """Remove a model from the registry. Returns True if it existed."""
        with cls._lock:
            if name in cls._models:
                del cls._models[name]
                return True
            return False

    @classmethod
    def exists(cls, name: str) -> bool:
        """Check if a model is registered."""
        with cls._lock:
            return name in cls._models

    @classmethod
    def get_info(cls) -> list[dict]:
        """Get detailed info about all loaded models.

        Returns a list of dicts with name, device, dtype, loaded_at,
        and age in seconds. Useful for the /models/info endpoint.
        """
        with cls._lock:
            now = time.time()
            return [
                {
                    "name": name,
                    "device": info.get("device", "unknown"),
                    "dtype": info.get("dtype", "unknown"),
                    "path": info.get("path", ""),
                    "loaded_at": info.get("loaded_at"),
                    "age_seconds": now - info.get("loaded_at", now),
                }
                for name, info in cls._models.items()
            ]

    @classmethod
    def clear(cls) -> int:
        """Remove all models from the registry. Returns count removed.

        Useful for graceful shutdown or testing.
        """
        with cls._lock:
            count = len(cls._models)
            cls._models.clear()
            return count
