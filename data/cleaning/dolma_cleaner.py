"""
Dolma-inspired data cleaning pipeline.
Dolma (Allen AI) combines multiple taggers:
- Language identification
- Quality heuristics (Gopher-style)
- Toxicity filtering
- Deduplication
- PII redaction
Reference: https://arxiv.org/abs/2402.00159
"""

import hashlib
import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from data.cleaning.quality_filters import (
    ContentFilter,
    QualityConfig,
    TextQualityFilter,
)


@dataclass
class DolmaConfig:
    # Gopher-style quality thresholds
    min_length: int = 200
    max_length: int = 100000
    min_words: int = 50
    max_words: int = 100000
    min_mean_word_length: float = 3.0
    max_mean_word_length: float = 10.0

    # Symbol ratios (Gopher)
    max_symbol_word_ratio: float = 0.1
    max_bullet_lines_ratio: float = 0.9
    max_ellipsis_lines_ratio: float = 0.3

    # Repetition (Gopher)
    max_duplicate_line_char_frac: float = 0.2
    max_duplicate_para_char_frac: float = 0.2
    max_top_ngram_char_frac: float = 0.20
    ngram_sizes: list = field(default_factory=lambda: [2, 3, 4])

    # Language
    allowed_languages: list = field(default_factory=lambda: ["en"])
    min_language_score: float = 0.65

    # Toxicity
    toxicity_threshold: float = 0.5
    enable_toxicity_filter: bool = True

    # PII
    redact_pii: bool = True

    # Output
    output_dir: str = "./data/processed/dolma"
    source_name: str = "web"


class GopherQualityFilter:
    """
    Gopher-style quality heuristics (DeepMind, 2021).
    Used as core quality signal in Dolma.
    
    Filters based on:
    - Word count and length statistics
    - Symbol/bullet/ellipsis ratios
    - Duplicate line/paragraph detection
    - Top n-gram repetition
    """

    def __init__(self, config: DolmaConfig):
        self.config = config

    def _tokenize_words(self, text: str) -> list[str]:
        return text.split()

    def _get_lines(self, text: str) -> list[str]:
        return [l.strip() for l in text.split('\n') if l.strip()]

    def _get_paragraphs(self, text: str) -> list[str]:
        return [
            p.strip()
            for p in re.split(r'\n\n+', text)
            if p.strip()
        ]

    def check_word_count(
        self, words: list[str]
    ) -> tuple[bool, str]:
        n = len(words)
        if n < self.config.min_words:
            return False, f"Too few words: {n}"
        if n > self.config.max_words:
            return False, f"Too many words: {n}"
        return True, ""

    def check_mean_word_length(
        self, words: list[str]
    ) -> tuple[bool, str]:
        if not words:
            return False, "No words"
        avg = sum(len(w) for w in words) / len(words)
        if avg < self.config.min_mean_word_length:
            return False, f"Mean word length too short: {avg:.2f}"
        if avg > self.config.max_mean_word_length:
            return False, f"Mean word length too long: {avg:.2f}"
        return True, ""

    def check_symbol_word_ratio(
        self, text: str, words: list[str]
    ) -> tuple[bool, str]:
        if not words:
            return False, "No words"
        symbols = sum(
            1 for c in text
            if c in '#<>{}/\\|@$%^&*~`'
        )
        ratio = symbols / len(words)
        if ratio > self.config.max_symbol_word_ratio:
            return False, f"High symbol/word ratio: {ratio:.3f}"
        return True, ""

    def check_bullet_lines(
        self, lines: list[str]
    ) -> tuple[bool, str]:
        if not lines:
            return True, ""
        bullet_count = sum(
            1 for l in lines
            if l.startswith(('•', '-', '*', '·', '◦', '▪', '–'))
        )
        ratio = bullet_count / len(lines)
        if ratio > self.config.max_bullet_lines_ratio:
            return False, f"High bullet ratio: {ratio:.3f}"
        return True, ""

    def check_ellipsis_lines(
        self, lines: list[str]
    ) -> tuple[bool, str]:
        if not lines:
            return True, ""
        ellipsis_count = sum(
            1 for l in lines
            if l.endswith('...') or '…' in l
        )
        ratio = ellipsis_count / len(lines)
        if ratio > self.config.max_ellipsis_lines_ratio:
            return False, f"High ellipsis ratio: {ratio:.3f}"
        return True, ""

    def check_duplicate_lines(
        self, lines: list[str], text: str
    ) -> tuple[bool, str]:
        if not lines or not text:
            return True, ""
        from collections import Counter
        counts = Counter(lines)
        dup_chars = sum(
            len(line) * (count - 1)
            for line, count in counts.items()
            if count > 1
        )
        ratio = dup_chars / max(len(text), 1)
        if ratio > self.config.max_duplicate_line_char_frac:
            return False, f"High dup line char frac: {ratio:.3f}"
        return True, ""

    def check_duplicate_paragraphs(
        self, paragraphs: list[str], text: str
    ) -> tuple[bool, str]:
        if not paragraphs or not text:
            return True, ""
        from collections import Counter
        counts = Counter(paragraphs)
        dup_chars = sum(
            len(p) * (count - 1)
            for p, count in counts.items()
            if count > 1
        )
        ratio = dup_chars / max(len(text), 1)
        if ratio > self.config.max_duplicate_para_char_frac:
            return False, f"High dup para char frac: {ratio:.3f}"
        return True, ""

    def check_top_ngram_repetition(
        self, words: list[str], text: str
    ) -> tuple[bool, str]:
        """
        Check if top n-grams dominate the text.
        Catches keyword-stuffed, repetitive content.
        """
        from collections import Counter
        for n in self.config.ngram_sizes:
            if len(words) < n:
                continue
            ngrams = [
                tuple(words[i:i+n])
                for i in range(len(words) - n + 1)
            ]
            counts = Counter(ngrams)
            if not counts:
                continue
            top_ngram, top_count = counts.most_common(1)[0]
            top_chars = len(' '.join(top_ngram)) * top_count
            ratio = top_chars / max(len(text), 1)
            if ratio > self.config.max_top_ngram_char_frac:
                return (
                    False,
                    f"Top {n}-gram dominates: "
                    f"'{' '.join(top_ngram)}' "
                    f"ratio={ratio:.3f}"
                )
        return True, ""

    def filter(
        self, text: str
    ) -> tuple[bool, float, list[str]]:
        """
        Run all Gopher filters.
        Returns (passed, quality_score, failure_reasons).
        """
        words = self._tokenize_words(text)
        lines = self._get_lines(text)
        paragraphs = self._get_paragraphs(text)

        checks = [
            self.check_word_count(words),
            self.check_mean_word_length(words),
            self.check_symbol_word_ratio(text, words),
            self.check_bullet_lines(lines),
            self.check_ellipsis_lines(lines),
            self.check_duplicate_lines(lines, text),
            self.check_duplicate_paragraphs(paragraphs, text),
            self.check_top_ngram_repetition(words, text),
        ]

        failures = [reason for passed, reason in checks if not passed]
        passed = len(failures) == 0
        score = (
            len(checks) - len(failures)
        ) / len(checks)

        return passed, score, failures


class LanguageTagger:
    """
    Language identification tagger.
    Uses fasttext-langdetect with fallback to langdetect.
    """

    def __init__(self, min_score: float = 0.65):
        self.min_score = min_score
        self._model = None
        self._backend = 'langdetect'
        self._detection_available = False
        self._load_model()

    def _load_model(self):
        try:
            import fasttext
            # fasttext lid model needed
            self._backend = 'fasttext'
            if self._model is not None:
                self._detection_available = True
        except ImportError:
            self._backend = 'langdetect'
            try:
                from langdetect import detect_langs
                self._detection_available = True
                logger.warning(
                    "fasttext not available, using langdetect"
                )
            except Exception:
                self._detection_available = False
                logger.warning(
                    "langdetect not available; disabling language filter"
                )

    def detect(
        self, text: str
    ) -> tuple[str, float]:
        """Returns (language_code, confidence)."""
        if not self._detection_available:
            return 'unknown', 0.0
        sample = text[:1000]
        try:
            if self._backend == 'fasttext' and self._model:
                labels, scores = self._model.predict(
                    sample.replace('\n', ' ')
                )
                lang = labels[0].replace('__label__', '')
                return lang, float(scores[0])
            else:
                from langdetect import detect_langs
                results = detect_langs(sample)
                if results:
                    top = results[0]
                    return top.lang, top.prob
        except Exception:
            pass
        return 'unknown', 0.0

    def is_allowed(
        self,
        text: str,
        allowed: list[str]
    ) -> tuple[bool, str, float]:
        if not self._detection_available:
            return True, 'unknown', 0.0
        lang, score = self.detect(text)
        if lang not in allowed:
            return (
                False,
                f"Wrong language: {lang} ({score:.2f})",
                score
            )
        if score < self.min_score:
            return (
                False,
                f"Low lang confidence: {score:.2f}",
                score
            )
        return True, lang, score


class ToxicityFilter:
    """
    Heuristic toxicity filter.
    In production, replace with a trained classifier.
    """

    # Simplified toxic keyword patterns
    TOXIC_PATTERNS = [
        r'\b(hate|kill|murder)\s+(all|every)\s+\w+',
        r'\b(racist|sexist|homophobic)\b.*\b(slur|attack)\b',
    ]

    NSFW_DOMAINS = {
        'xxx', 'porn', 'adult', 'nsfw', 'sex'
    }

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.patterns = [
            re.compile(p, re.IGNORECASE)
            for p in self.TOXIC_PATTERNS
        ]

    def score(self, text: str) -> float:
        """
        Returns toxicity score 0-1.
        0 = clean, 1 = toxic.
        """
        hits = sum(
            1 for p in self.patterns if p.search(text)
        )
        return min(1.0, hits / max(len(self.patterns), 1))

    def is_toxic(self, text: str) -> tuple[bool, float]:
        score = self.score(text)
        return score >= self.threshold, score

    def check_url(self, url: str) -> bool:
        """Check if URL domain suggests NSFW content."""
        url_lower = url.lower()
        return any(
            domain in url_lower
            for domain in self.NSFW_DOMAINS
        )


class PIIRedactor:
    """
    PII (Personally Identifiable Information) redactor.
    Finds and replaces sensitive information.
    """

    PATTERNS = {
        'ssn': (
            re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            '[SSN_REDACTED]'
        ),
        'credit_card': (
            re.compile(r'\b(?:\d{4}[- ]?){3}\d{4}\b'),
            '[CC_REDACTED]'
        ),
        'email': (
            re.compile(
                r'\b[A-Za-z0-9._%+-]+'
                r'@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ),
            '[EMAIL_REDACTED]'
        ),
        'phone_us': (
            re.compile(
                r'\b(?:\+?1[-.\s]?)?'
                r'(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b'
            ),
            '[PHONE_REDACTED]'
        ),
        'ip_address': (
            re.compile(
                r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
            ),
            '[IP_REDACTED]'
        ),
        'passport': (
            re.compile(r'\b[A-Z]{1,2}\d{6,9}\b'),
            '[PASSPORT_REDACTED]'
        ),
    }

    def has_pii(self, text: str) -> bool:
        return any(
            pattern.search(text)
            for pattern, _ in self.PATTERNS.values()
        )

    def redact(self, text: str) -> tuple[str, list[str]]:
        """Redact all PII and return redacted text + types found."""
        found_types = []
        for pii_type, (pattern, replacement) in self.PATTERNS.items():
            if pattern.search(text):
                found_types.append(pii_type)
                text = pattern.sub(replacement, text)
        return text, found_types


class DolmaTextCleaner:
    """
    Full Dolma-style cleaning pipeline.
    
    Stages (matching Dolma v1.7 pipeline):
    1. URL-based filtering
    2. Gopher quality heuristics
    3. Language identification
    4. Toxicity filtering
    5. PII redaction
    6. Exact deduplication
    7. Text normalization
    
    Reference: Soldaini et al. (2024) Dolma: an Open Corpus
    of Three Trillion Tokens for Language Model Pretraining
    """

    def __init__(self, config: DolmaConfig):
        self.config = config
        self.gopher = GopherQualityFilter(config)
        self.lang_tagger = LanguageTagger(config.min_language_score)
        self.toxicity = ToxicityFilter(config.toxicity_threshold)
        self.pii_redactor = PIIRedactor()
        self.content_filter = ContentFilter()
        self.seen_hashes: set[str] = set()

        os.makedirs(config.output_dir, exist_ok=True)

        self.stats = {
            'total': 0,
            'passed': 0,
            'failed_gopher': 0,
            'failed_language': 0,
            'failed_toxicity': 0,
            'failed_exact_dedup': 0,
            'pii_redacted': 0,
        }

    def _normalize_text(self, text: str) -> str:
        """Unicode normalization and whitespace cleanup."""
        text = unicodedata.normalize('NFC', text)
        text = re.sub(r'\r\n|\r', '\n', text)
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r' \n', '\n', text)
        return text.strip()

    def _exact_dedup(self, text: str) -> bool:
        """Return True if duplicate (should skip)."""
        h = hashlib.sha256(text.encode()).hexdigest()
        if h in self.seen_hashes:
            return True
        self.seen_hashes.add(h)
        return False

    def _tag_document(self, doc: dict) -> dict:
        """Add quality tags to document without filtering."""
        text = doc.get('text', '')
        _, gopher_score, gopher_fails = self.gopher.filter(text)
        lang, lang_conf = self.lang_tagger.detect(text)
        tox_score = self.toxicity.score(text)
        has_pii = self.pii_redactor.has_pii(text)

        return {
            **doc,
            'tags': {
                'gopher_score': gopher_score,
                'gopher_failures': gopher_fails,
                'language': lang,
                'language_confidence': lang_conf,
                'toxicity_score': tox_score,
                'has_pii': has_pii,
                'source': self.config.source_name,
            }
        }

    def process_document(
        self, doc: dict
    ) -> Optional[dict]:
        self.stats['total'] += 1
        text = doc.get('text', '')

        if not text or len(text) < self.config.min_length:
            return None

        # ── Stage 1: Normalize ─────────────────────────────────
        text = self._normalize_text(text)

        # ── Stage 2: Gopher Quality ────────────────────────────
        gopher_passed, gopher_score, failures = (
            self.gopher.filter(text)
        )
        if not gopher_passed:
            self.stats['failed_gopher'] += 1
            logger.debug(f"Gopher fail: {failures[:2]}")
            return None

        # ── Stage 3: Language Filter ───────────────────────────
        lang_ok, lang_info, lang_conf = self.lang_tagger.is_allowed(
            text, self.config.allowed_languages
        )
        if not lang_ok:
            self.stats['failed_language'] += 1
            return None

        # ── Stage 4: Toxicity Filter ───────────────────────────
        if self.config.enable_toxicity_filter:
            is_toxic, tox_score = self.toxicity.is_toxic(text)
            if is_toxic:
                self.stats['failed_toxicity'] += 1
                return None

            # Also check source URL
            url = doc.get('url', '')
            if url and self.toxicity.check_url(url):
                self.stats['failed_toxicity'] += 1
                return None

        # ── Stage 5: PII Redaction ─────────────────────────────
        pii_types = []
        if self.config.redact_pii:
            text, pii_types = self.pii_redactor.redact(text)
            if pii_types:
                self.stats['pii_redacted'] += 1

        # ── Stage 6: Exact Deduplication ──────────────────────
        if self._exact_dedup(text):
            self.stats['failed_exact_dedup'] += 1
            return None

        self.stats['passed'] += 1

        return {
            **doc,
            'text': text,
            'dolma_metadata': {
                'gopher_score': gopher_score,
                'language': lang_info,
                'language_confidence': lang_conf,
                'pii_types_redacted': pii_types,
                'source': self.config.source_name,
                'pipeline': 'dolma',
            }
        }

    def process_file(
        self,
        input_file: str,
        output_file: str
    ) -> dict:
        """Process a JSONL file through the full Dolma pipeline."""
        logger.info(f"Dolma processing: {input_file}")

        with (
            open(input_file, 'r', encoding='utf-8') as fin,
            open(output_file, 'w', encoding='utf-8') as fout
        ):
            for line_num, line in enumerate(fin):
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                    cleaned = self.process_document(doc)
                    if cleaned:
                        fout.write(json.dumps(cleaned) + '\n')
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON error line {line_num}: {e}")
                except Exception as e:
                    logger.error(f"Error line {line_num}: {e}")

                if line_num % 10000 == 0 and line_num > 0:
                    logger.info(
                        f"Dolma progress: {line_num} | "
                        f"Stats: {self.stats}"
                    )

        logger.info(f"Dolma complete. Stats: {self.stats}")
        return self.stats

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        pattern: str = "*.jsonl"
    ) -> dict:
        """Process all JSONL files in a directory."""
        import glob
        os.makedirs(output_dir, exist_ok=True)
        files = glob.glob(os.path.join(input_dir, pattern))
        logger.info(f"Processing {len(files)} files")

        for fpath in files:
            fname = os.path.basename(fpath)
            out_path = os.path.join(output_dir, fname)
            self.process_file(fpath, out_path)

        return self.stats