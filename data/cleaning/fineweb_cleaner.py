"""
FineWeb-inspired cleaning pipeline.
Combines quality signals with educational content scoring.
"""

import json
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
from loguru import logger

from data.cleaning.quality_filters import TextQualityFilter, QualityConfig


@dataclass
class FineWebConfig:
    quality: QualityConfig = None
    educational_threshold: float = 0.3
    min_educational_score: float = 0.2

    def __post_init__(self):
        if self.quality is None:
            self.quality = QualityConfig(
                min_length=200,
                min_words=50,
                min_unique_words_ratio=0.3
            )


class EducationalScorer:
    """
    Heuristic educational quality scorer.
    Inspired by FineWeb-Edu's approach.
    """

    EDUCATIONAL_KEYWORDS = {
        'high': [
            'therefore', 'consequently', 'furthermore', 'however',
            'analysis', 'evidence', 'hypothesis', 'conclusion',
            'research', 'study', 'theory', 'concept', 'principle',
            'definition', 'example', 'explain', 'understand',
            'learn', 'knowledge', 'science', 'mathematics',
            'history', 'literature', 'philosophy'
        ],
        'medium': [
            'because', 'although', 'despite', 'while', 'since',
            'show', 'demonstrate', 'describe', 'discuss',
            'important', 'significant', 'useful', 'helpful'
        ],
        'low': [
            'click', 'subscribe', 'buy', 'sale', 'discount',
            'follow', 'like', 'share', 'comment', 'tweet',
            'cookie', 'privacy policy', 'terms of service'
        ]
    }

    STRUCTURE_PATTERNS = {
        'paragraph': re.compile(r'\n\n'),
        'numbered_list': re.compile(r'^\d+[\.\)]\s', re.MULTILINE),
        'sentence_end': re.compile(r'[.!?]'),
        'long_sentence': re.compile(r'[^.!?]{100,}[.!?]'),
    }

    def __init__(self):
        self.high_kws = set(self.EDUCATIONAL_KEYWORDS['high'])
        self.medium_kws = set(self.EDUCATIONAL_KEYWORDS['medium'])
        self.low_kws = set(self.EDUCATIONAL_KEYWORDS['low'])

    def score(self, text: str) -> float:
        words_lower = set(text.lower().split())
        total_words = len(text.split())

        if total_words == 0:
            return 0.0

        high_count = len(words_lower & self.high_kws)
        medium_count = len(words_lower & self.medium_kws)
        low_count = len(words_lower & self.low_kws)

        keyword_score = (
            high_count * 2.0 + medium_count * 1.0 - low_count * 3.0
        ) / max(total_words, 1)

        paragraphs = len(
            self.STRUCTURE_PATTERNS['paragraph'].findall(text)
        )
        sentences = len(
            self.STRUCTURE_PATTERNS['sentence_end'].findall(text)
        )
        structure_score = min(
            1.0, (paragraphs * 0.1 + sentences * 0.02)
        )

        avg_sentence_len = total_words / max(sentences, 1)
        length_score = (
            1.0 if 10 <= avg_sentence_len <= 30 else 0.5
        )

        score = (
            keyword_score * 0.4 +
            structure_score * 0.4 +
            length_score * 0.2
        )
        return float(np.clip(score, 0, 1))


class FineWebCleaner:
    """FineWeb-style cleaning with educational quality scoring."""

    def __init__(self, config: FineWebConfig):
        self.config = config
        self.quality_filter = TextQualityFilter(config.quality)
        self.edu_scorer = EducationalScorer()
        self.stats = {
            'total': 0,
            'passed': 0,
            'failed_quality': 0,
            'failed_educational': 0,
        }

    def process_document(
        self,
        doc: dict
    ) -> Optional[dict]:
        self.stats['total'] += 1
        text = doc.get('text', '')

        if not text:
            return None

        quality_result = self.quality_filter.filter(text)
        if not quality_result.passed:
            self.stats['failed_quality'] += 1
            return None

        edu_score = self.edu_scorer.score(text)
        if edu_score < self.config.min_educational_score:
            self.stats['failed_educational'] += 1
            return None

        self.stats['passed'] += 1
        return {
            **doc,
            'quality_score': quality_result.score,
            'educational_score': edu_score,
            'source_pipeline': 'fineweb'
        }

    def process_file(
        self,
        input_file: str,
        output_file: str
    ) -> dict:
        with (
            open(input_file, 'r', encoding='utf-8') as fin,
            open(output_file, 'w', encoding='utf-8') as fout
        ):
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                    cleaned = self.process_document(doc)
                    if cleaned:
                        fout.write(json.dumps(cleaned) + '\n')
                except Exception as e:
                    logger.error(f"Error: {e}")

        logger.info(f"FineWeb stats: {self.stats}")
        return self.stats