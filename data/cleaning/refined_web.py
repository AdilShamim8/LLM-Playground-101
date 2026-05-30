"""
RefinedWeb-inspired data cleaning pipeline.
Implements quality filtering, deduplication (MinHash + exact),
and language filtering.
"""

import hashlib
import json
import os
import re
import struct
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from loguru import logger

from data.cleaning.quality_filters import (
    ContentFilter,
    QualityConfig,
    TextQualityFilter,
)


@dataclass
class RefinedWebConfig:
    quality: QualityConfig = None
    minhash_num_perm: int = 128
    minhash_threshold: float = 0.85
    ngram_size: int = 5
    exact_dedup: bool = True
    near_dedup: bool = True
    output_dir: str = "./data/processed/refined_web"

    def __post_init__(self):
        if self.quality is None:
            self.quality = QualityConfig()


class MinHashDeduplicator:
    """
    MinHash LSH-based near-duplicate detection.
    Efficient O(n) approximate deduplication.
    """

    _MERSENNE_PRIME = (1 << 61) - 1
    _MAX_HASH = (1 << 32) - 1

    def __init__(self, num_perm: int = 128, threshold: float = 0.85):
        self.num_perm = num_perm
        self.threshold = threshold
        self._init_hash_params()
        self.buckets: dict[tuple, list] = defaultdict(list)
        self.band_size = self._compute_band_size()
        self.num_bands = num_perm // self.band_size

    def _init_hash_params(self):
        rng = np.random.RandomState(42)
        self.a = rng.randint(
            1, self._MERSENNE_PRIME, size=self.num_perm, dtype=np.int64
        )
        self.b = rng.randint(
            0, self._MERSENNE_PRIME, size=self.num_perm, dtype=np.int64
        )

    def _compute_band_size(self) -> int:
        # b * r = n, threshold ≈ (1/b)^(1/r)
        # We use simple heuristic: band_size = 4
        return 4

    def _hash_value(self, value: int) -> np.ndarray:
        hashes = (self.a * value + self.b) % self._MERSENNE_PRIME
        return hashes % self._MAX_HASH

    def compute_minhash(self, text: str) -> np.ndarray:
        """Compute MinHash signature for text."""
        tokens = self._get_ngrams(text)
        signature = np.full(self.num_perm, self._MAX_HASH, dtype=np.int64)

        for token in tokens:
            token_hash = int(
                hashlib.md5(token.encode()).hexdigest(), 16
            ) & self._MAX_HASH
            hashes = self._hash_value(token_hash)
            signature = np.minimum(signature, hashes)

        return signature

    def _get_ngrams(self, text: str, n: int = 5) -> set[str]:
        words = text.lower().split()
        if len(words) < n:
            return set(words)
        return {
            ' '.join(words[i:i+n])
            for i in range(len(words) - n + 1)
        }

    def is_duplicate(
        self,
        doc_id: str,
        signature: np.ndarray
    ) -> bool:
        """
        Check if document is near-duplicate of any seen document.
        Uses LSH banding technique.
        """
        is_dup = False
        bands = []

        for band_idx in range(self.num_bands):
            start = band_idx * self.band_size
            end = start + self.band_size
            band = tuple(signature[start:end].tolist())
            bucket_key = (band_idx, band)
            bands.append(bucket_key)

            if self.buckets[bucket_key]:
                is_dup = True
                break

        if not is_dup:
            for bucket_key in bands:
                self.buckets[bucket_key].append(doc_id)

        return is_dup


class ExactDeduplicator:
    """Exact deduplication using SHA-256 hashes."""

    def __init__(self):
        self.seen_hashes: set[str] = set()

    def is_duplicate(self, text: str) -> bool:
        h = hashlib.sha256(text.encode()).hexdigest()
        if h in self.seen_hashes:
            return True
        self.seen_hashes.add(h)
        return False

    def __len__(self):
        return len(self.seen_hashes)


class RefinedWebCleaner:
    """
    Full RefinedWeb-style cleaning pipeline:
    1. URL filtering
    2. Text extraction quality
    3. Quality heuristics
    4. Language filtering
    5. Exact deduplication
    6. Near-duplicate detection (MinHash)
    """

    def __init__(self, config: RefinedWebConfig):
        self.config = config
        self.quality_filter = TextQualityFilter(config.quality)
        self.content_filter = ContentFilter()
        self.exact_dedup = ExactDeduplicator()
        self.minhash_dedup = MinHashDeduplicator(
            num_perm=config.minhash_num_perm,
            threshold=config.minhash_threshold
        )
        os.makedirs(config.output_dir, exist_ok=True)

        self.stats = {
            'total': 0,
            'passed': 0,
            'failed_quality': 0,
            'failed_content': 0,
            'failed_exact_dedup': 0,
            'failed_near_dedup': 0,
        }

    def clean_text(self, text: str) -> str:
        """Basic text normalization."""
        text = unicodedata.normalize('NFC', text)
        text = re.sub(r'\r\n|\r', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(
            r'[^\x09\x0A\x0D\x20-\x7E\x80-\xFF]', '', text
        )
        return text.strip()

    def process_document(
        self,
        doc: dict,
        doc_id: str
    ) -> Optional[dict]:
        self.stats['total'] += 1
        text = doc.get('text', '')

        if not text:
            return None

        text = self.clean_text(text)

        # Quality filter
        quality_result = self.quality_filter.filter(text)
        if not quality_result.passed:
            self.stats['failed_quality'] += 1
            logger.debug(
                f"Quality fail [{doc_id}]: {quality_result.reasons}"
            )
            return None

        # Content safety filter
        is_safe, issues = self.content_filter.is_safe(text)
        if not is_safe:
            self.stats['failed_content'] += 1
            if 'pii_detected' in issues:
                text = self.content_filter.redact_pii(text)
            else:
                return None

        # Exact deduplication
        if self.config.exact_dedup:
            if self.exact_dedup.is_duplicate(text):
                self.stats['failed_exact_dedup'] += 1
                return None

        # Near-duplicate detection
        if self.config.near_dedup:
            signature = self.minhash_dedup.compute_minhash(text)
            if self.minhash_dedup.is_duplicate(doc_id, signature):
                self.stats['failed_near_dedup'] += 1
                return None

        self.stats['passed'] += 1
        return {
            **doc,
            'text': text,
            'quality_score': quality_result.score,
            'cleaned': True
        }

    def process_file(
        self,
        input_file: str,
        output_file: str
    ) -> dict:
        """Process a JSONL input file and write cleaned output."""
        logger.info(f"Processing: {input_file}")

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
                    doc_id = doc.get(
                        'content_hash',
                        hashlib.md5(line.encode()).hexdigest()
                    )
                    cleaned = self.process_document(doc, doc_id)
                    if cleaned:
                        fout.write(json.dumps(cleaned) + '\n')
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON error line {line_num}: {e}")
                except Exception as e:
                    logger.error(f"Error line {line_num}: {e}")

                if line_num % 10000 == 0:
                    logger.info(
                        f"Progress: {line_num} lines | "
                        f"Stats: {self.stats}"
                    )

        logger.info(f"Cleaning stats: {self.stats}")
        return self.stats