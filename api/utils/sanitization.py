"""
Input sanitization utilities.

WHY THIS EXISTS:
User-provided text (prompts, system messages) can contain:
- Prompt injection attacks ("ignore previous instructions and...")
- Control characters (null bytes, escape sequences)
- Excessively long inputs (DoS)
- HTML/script injection (if output is rendered in a browser)

These utilities provide basic sanitization. For production, also add:
- Prompt injection detection (use a classifier model)
- Content moderation (toxicity filter)
- Rate limiting per user
"""

import re
from typing import Optional


def sanitize_prompt(text: str, max_length: int = 32000) -> str:
    """Sanitize user-provided prompt text.

    Args:
        text: Raw user input
        max_length: Maximum allowed length (default 32K chars)

    Returns:
        Sanitized text with control characters removed and length capped.
    """
    if not text:
        return ""

    # Remove null bytes and other control characters (except newline, tab)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

    # Normalize unicode (NFC form)
    text = text.strip()

    # Truncate to max length
    if len(text) > max_length:
        text = text[:max_length]

    return text


def sanitize_model_name(name: str) -> str:
    """Sanitize model names to prevent path traversal.

    Ensures the name only contains alphanumeric chars, hyphens, underscores.
    Prevents attacks like: model_name = "../../../etc/passwd"
    """
    if not name:
        return ""
    # Only allow alphanumeric, hyphen, underscore, dot
    sanitized = re.sub(r"[^a-zA-Z0-9._-]", "", name)
    # Prevent path traversal
    sanitized = sanitized.replace("..", "")
    return sanitized


def detect_prompt_injection(text: str) -> tuple[bool, Optional[str]]:
    """Basic prompt injection detection.

    Returns (is_injection, reason) tuple. This is a simple heuristic
    check — for production, use a trained classifier model.

    Checks for common injection patterns:
    - "ignore previous instructions"
    - "you are now a DAN" (Do Anything Now)
    - "system prompt:" / "system:" at the start
    - Attempts to override role
    """
    text_lower = text.lower().strip()

    injection_patterns = [
        ("ignore previous instructions", "ignore instruction attempt"),
        ("ignore all previous", "ignore instruction attempt"),
        ("ignore your instructions", "ignore instruction attempt"),
        ("you are now dan", "DAN jailbreak attempt"),
        ("do anything now", "DAN jailbreak attempt"),
        ("act as if you have no restrictions", "restriction bypass attempt"),
        ("pretend you are", "role override attempt"),
        ("system: you are", "system prompt injection"),
        ("<|system|>", "special token injection"),
        ("<|im_start|>system", "chatml injection"),
    ]

    for pattern, reason in injection_patterns:
        if pattern in text_lower:
            return True, reason

    return False, None


def truncate_to_token_limit(
    text: str,
    tokenizer,
    max_tokens: int = 4096,
) -> str:
    """Truncate text to fit within a token limit.

    Args:
        text: Input text
        tokenizer: Tokenizer with encode/decode methods
        max_tokens: Maximum tokens to keep

    Returns:
        Truncated text that fits within the token limit.
    """
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text

    # Keep the first max_tokens tokens (most context is at the start)
    truncated_tokens = tokens[:max_tokens]
    return tokenizer.decode(truncated_tokens)
