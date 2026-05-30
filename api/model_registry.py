"""
Shared model registry — single source of truth.
Extracted from chat.py to break circular import
between chat.py and models.py.
"""

import time
from typing import Optional


class ModelRegistry:
    """
    Thread-safe in-memory model registry.
    Stores loaded models, tokenizers, and generators.
    Use Redis/DB backend in production for multi-worker setups.
    """

    _models: dict[str, dict] = {}

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
    ):
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
        return cls._models.get(name)

    @classmethod
    def list_models(cls) -> list[str]:
        return list(cls._models.keys())

    @classmethod
    def unregister(cls, name: str) -> bool:
        if name in cls._models:
            del cls._models[name]
            return True
        return False

    @classmethod
    def exists(cls, name: str) -> bool:
        return name in cls._models