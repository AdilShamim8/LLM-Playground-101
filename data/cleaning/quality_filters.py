"""
Quality filters for web-crawled text data.
Implements heuristic and ML-based quality filtering.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

from loguru import logger


@dataclass
class QualityConfig:
    min_length: int = 100
    max_length: int = 100000
    min_words: int = 20
    max_words: int = 100000
    min_avg_word_length: float = 3.0
    max_avg_word_length: float = 10.0
    max_symbol_ratio: float = 0.1
    max_digit_ratio: float = 0.2
    max_uppercase_ratio: float = 0.3
    max_line_repeat_ratio: float = 0.3
    min_unique_words_ratio: float = 0.2
    max_bullet_ratio: float = 0.9
    allowed_languages: list = None

    def __post_init__(self):
        if self.allowed_languages is None:
            self.allowed_languages = ['en']


@dataclass
class FilterResult:
    passed: bool
    score: float
    reasons: list[str]


class TextQualityFilter:
    """
    Multi-stage quality filter inspired by RefinedWeb, Dolma, FineWeb.
    Filters low-quality, noisy, and harmful text.
    """

    def __init__(self, config: QualityConfig):
        self.config = config
        self._compile_patterns()

    def _compile_patterns(self):
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|'
            r'(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        self.html_pattern = re.compile(r'<[^>]+>')
        self.whitespace_pattern = re.compile(r'\s+')
        self.special_chars = re.compile(
            r'[^\w\s\.\,\!\?\;\:\'\"\-\(\)\[\]\{\}]'
        )

    def check_length(self, text: str) -> tuple[bool, str]:
        length = len(text)
        if length < self.config.min_length:
            return False, f"Too short: {length} chars"
        if length > self.config.max_length:
            return False, f"Too long: {length} chars"
        return True, ""

    def check_word_count(self, words: list[str]) -> tuple[bool, str]:
        count = len(words)
        if count < self.config.min_words:
            return False, f"Too few words: {count}"
        if count > self.config.max_words:
            return False, f"Too many words: {count}"
        return True, ""

    def check_word_length(
        self, words: list[str]
    ) -> tuple[bool, str]:
        if not words:
            return False, "No words"
        avg_len = sum(len(w) for w in words) / len(words)
        if avg_len < self.config.min_avg_word_length:
            return False, f"Avg word too short: {avg_len:.2f}"
        if avg_len > self.config.max_avg_word_length:
            return False, f"Avg word too long: {avg_len:.2f}"
        return True, ""

    def check_symbol_ratio(
        self, text: str
    ) -> tuple[bool, str]:
        total = len(text)
        if total == 0:
            return False, "Empty"
        symbols = sum(
            1 for c in text
            if not c.isalnum() and not c.isspace()
        )
        ratio = symbols / total
        if ratio > self.config.max_symbol_ratio:
            return False, f"High symbol ratio: {ratio:.2f}"
        return True, ""

    def check_digit_ratio(
        self, text: str
    ) -> tuple[bool, str]:
        total = len(text)
        if total == 0:
            return False, "Empty"
        digits = sum(1 for c in text if c.isdigit())
        ratio = digits / total
        if ratio > self.config.max_digit_ratio:
            return False, f"High digit ratio: {ratio:.2f}"
        return True, ""

    def check_uppercase_ratio(
        self, text: str
    ) -> tuple[bool, str]:
        alpha = [c for c in text if c.isalpha()]
        if not alpha:
            return False, "No alphabetic chars"
        upper = sum(1 for c in alpha if c.isupper())
        ratio = upper / len(alpha)
        if ratio > self.config.max_uppercase_ratio:
            return False, f"High uppercase ratio: {ratio:.2f}"
        return True, ""

    def check_line_repetition(
        self, text: str
    ) -> tuple[bool, str]:
        lines = text.split('\n')
        if not lines:
            return True, ""
        line_counts: dict[str, int] = {}
        for line in lines:
            stripped = line.strip()
            if stripped:
                line_counts[stripped] = (
                    line_counts.get(stripped, 0) + 1
                )

        total_lines = len([l for l in lines if l.strip()])
        if total_lines <= 1:
            return True, ""

        max_repeats = max(line_counts.values()) if line_counts else 0
        ratio = max_repeats / total_lines

        if ratio > self.config.max_line_repeat_ratio:
            return False, f"High line repeat ratio: {ratio:.2f}"
        return True, ""

    def check_unique_words(
        self, words: list[str]
    ) -> tuple[bool, str]:
        if not words:
            return False, "No words"
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < self.config.min_unique_words_ratio:
            return False, f"Low unique ratio: {unique_ratio:.2f}"
        return True, ""

    def check_html_remnants(
        self, text: str
    ) -> tuple[bool, str]:
        html_count = len(self.html_pattern.findall(text))
        ratio = html_count / max(1, len(text.split()))
        if ratio > 0.05:
            return False, f"HTML remnants detected: {html_count}"
        return True, ""

    def check_bullet_ratio(
        self, text: str
    ) -> tuple[bool, str]:
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if not lines:
            return True, ""
        bullet_count = sum(
            1 for l in lines
            if l.startswith(('•', '-', '*', '·', '◦', '▪'))
        )
        ratio = bullet_count / len(lines)
        if ratio > self.config.max_bullet_ratio:
            return False, f"Too many bullets: {ratio:.2f}"
        return True, ""

    def detect_language(self, text: str) -> str:
        """Simple language detection."""
        try:
            from langdetect import detect
            return detect(text[:500])
        except Exception:
            return 'unknown'

    def compute_quality_score(
        self,
        text: str,
        words: list[str],
        checks_passed: int,
        total_checks: int
    ) -> float:
        base_score = checks_passed / total_checks
        length_bonus = min(
            1.0, len(words) / 500
        ) * 0.1
        unique_bonus = (
            len(set(words)) / max(1, len(words))
        ) * 0.1
        return min(1.0, base_score + length_bonus + unique_bonus)

    def filter(self, text: str) -> FilterResult:
        """Run all quality checks and return result."""
        reasons = []
        checks_passed = 0
        total_checks = 0

        words = self.whitespace_pattern.split(text.strip())
        words = [w for w in words if w]

        checks = [
            self.check_length(text),
            self.check_word_count(words),
            self.check_word_length(words),
            self.check_symbol_ratio(text),
            self.check_digit_ratio(text),
            self.check_uppercase_ratio(text),
            self.check_line_repetition(text),
            self.check_unique_words(words),
            self.check_html_remnants(text),
            self.check_bullet_ratio(text),
        ]

        for passed, reason in checks:
            total_checks += 1
            if passed:
                checks_passed += 1
            else:
                reasons.append(reason)

        passed = len(reasons) == 0
        score = self.compute_quality_score(
            text, words, checks_passed, total_checks
        )

        return FilterResult(
            passed=passed,
            score=score,
            reasons=reasons
        )


class ContentFilter:
    """
    Filters for harmful, private, or inappropriate content.
    """

    HATE_SPEECH_PATTERNS = [
        r'\b(kill|murder|attack)\s+(all|every)\b',
    ]

    PII_PATTERNS = [
        r'\b\d{3}-\d{2}-\d{4}\b',           # SSN
        r'\b\d{16}\b',                         # Credit card
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
    ]

    def __init__(self):
        self.hate_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in self.HATE_SPEECH_PATTERNS
        ]
        self.pii_patterns = [
            re.compile(p)
            for p in self.PII_PATTERNS
        ]

    def has_harmful_content(self, text: str) -> bool:
        return any(p.search(text) for p in self.hate_patterns)

    def has_pii(self, text: str) -> bool:
        return any(p.search(text) for p in self.pii_patterns)

    def redact_pii(self, text: str) -> str:
        for pattern in self.pii_patterns:
            text = pattern.sub('[REDACTED]', text)
        return text

    def is_safe(self, text: str) -> tuple[bool, list[str]]:
        issues = []
        if self.has_harmful_content(text):
            issues.append("harmful_content")
        if self.has_pii(text):
            issues.append("pii_detected")
        return len(issues) == 0, issues