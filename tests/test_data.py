"""Tests for data pipeline components."""

import json
import os
import tempfile
import pytest

from data.tokenization.bpe_tokenizer import BPEConfig, ByteLevelBPETokenizer
from data.cleaning.quality_filters import (
    QualityConfig, TextQualityFilter, ContentFilter
)
from data.cleaning.refined_web import (
    MinHashDeduplicator, ExactDeduplicator, RefinedWebConfig
)
from data.cleaning.dolma_cleaner import (
    DolmaConfig, GopherQualityFilter, PIIRedactor
)
from data.cleaning.fineweb_cleaner import (
    FineWebConfig, EducationalScorer
)


# ── Tokenizer Tests ───────────────────────────────────────────────

class TestBPETokenizer:

    def test_byte_encoder_coverage(self):
        config = BPEConfig(vocab_size=300)
        tok = ByteLevelBPETokenizer(config)
        assert len(tok.byte_encoder) == 256
        assert len(tok.byte_decoder) == 256

    def test_train_small_corpus(self):
        config = BPEConfig(
            vocab_size=300,
            min_frequency=1,
            special_tokens={
                'bos_token': '<|start|>',
                'eos_token': '<|end|>',
                'unk_token': '<|unk|>',
                'pad_token': '<|pad|>',
            }
        )
        tok = ByteLevelBPETokenizer(config)
        corpus = [
            "hello world this is a test sentence",
            "machine learning is fascinating",
            "natural language processing with transformers",
            "hello world again for testing the tokenizer",
        ] * 10
        tok.train(corpus)
        assert tok.vocab_size > 260

    def test_encode_decode_roundtrip(self):
        config = BPEConfig(
            vocab_size=300,
            min_frequency=1,
            special_tokens={
                'bos_token': '<|start|>',
                'eos_token': '<|end|>',
                'unk_token': '<|unk|>',
                'pad_token': '<|pad|>',
            }
        )
        tok = ByteLevelBPETokenizer(config)
        corpus = ["hello world test encode decode"] * 20
        tok.train(corpus)
        text = "hello world"
        ids = tok.encode(text, add_special_tokens=False)
        decoded = tok.decode(ids, skip_special_tokens=True)
        assert text in decoded or decoded in text

    def test_encode_batch(self):
        config = BPEConfig(
            vocab_size=300, min_frequency=1,
            special_tokens={
                'bos_token': '<|s|>', 'eos_token': '<|e|>',
                'unk_token': '<|u|>', 'pad_token': '<|p|>'
            }
        )
        tok = ByteLevelBPETokenizer(config)
        tok.train(["hello world test"] * 20)
        result = tok.encode_batch(
            ["hello", "hello world"],
            padding=True, max_length=32
        )
        assert 'input_ids' in result
        assert 'attention_mask' in result
        assert len(result['input_ids']) == 2
        assert len(result['input_ids'][0]) == len(
            result['input_ids'][1]
        )

    def test_save_load(self, tmp_path):
        config = BPEConfig(
            vocab_size=300, min_frequency=1,
            special_tokens={
                'bos_token': '<|s|>', 'eos_token': '<|e|>',
                'unk_token': '<|u|>', 'pad_token': '<|p|>'
            }
        )
        tok = ByteLevelBPETokenizer(config)
        tok.train(["hello world testing save load"] * 20)
        save_dir = str(tmp_path / "tokenizer")
        tok.save(save_dir)

        loaded = ByteLevelBPETokenizer.load(save_dir)
        assert loaded.vocab_size == tok.vocab_size
        assert set(loaded.encoder.keys()) == set(tok.encoder.keys())


# ── Quality Filter Tests ──────────────────────────────────────────

class TestQualityFilters:

    def test_good_text_passes(self):
        config = QualityConfig()
        filt = TextQualityFilter(config)
        good_text = (
            "Machine learning is a branch of artificial intelligence "
            "that focuses on developing systems that learn from data. "
            "It encompasses supervised, unsupervised, and reinforcement "
            "learning paradigms. Deep learning, a subset of machine "
            "learning, uses neural networks with many layers to model "
            "complex patterns in large datasets."
        )
        result = filt.filter(good_text)
        assert result.passed

    def test_short_text_fails(self):
        config = QualityConfig(min_length=100)
        filt = TextQualityFilter(config)
        result = filt.filter("Short text.")
        assert not result.passed

    def test_high_symbol_ratio_fails(self):
        config = QualityConfig(max_symbol_ratio=0.1)
        filt = TextQualityFilter(config)
        text = "#@$#@$#@$ " * 50 + " normal words here"
        result = filt.filter(text)
        assert not result.passed

    def test_content_filter_pii(self):
        cf = ContentFilter()
        text = "Contact me at user@example.com for info"
        is_safe, issues = cf.is_safe(text)
        assert "pii_detected" in issues

    def test_content_filter_redact(self):
        cf = ContentFilter()
        text = "SSN: 123-45-6789 and email: test@example.com"
        redacted = cf.redact_pii(text)
        assert "123-45-6789" not in redacted
        assert "test@example.com" not in redacted


# ── Deduplication Tests ───────────────────────────────────────────

class TestDeduplication:

    def test_exact_dedup(self):
        dedup = ExactDeduplicator()
        text = "This is a unique document for testing."
        assert not dedup.is_duplicate(text)
        assert dedup.is_duplicate(text)     # Second time = dup
        assert len(dedup) == 1

    def test_exact_dedup_different(self):
        dedup = ExactDeduplicator()
        assert not dedup.is_duplicate("Document A content here.")
        assert not dedup.is_duplicate("Document B different content.")
        assert len(dedup) == 2

    def test_minhash_exact_duplicate(self):
        mh = MinHashDeduplicator(num_perm=64, threshold=0.85)
        text = "The quick brown fox jumps over the lazy dog"
        sig = mh.compute_minhash(text)
        assert not mh.is_duplicate("doc1", sig)
        assert mh.is_duplicate("doc1_copy", sig)

    def test_minhash_different_docs(self):
        mh = MinHashDeduplicator(num_perm=64, threshold=0.85)
        text1 = "Machine learning is a powerful technology for data."
        text2 = "The history of Rome spans many centuries of civilization."
        sig1 = mh.compute_minhash(text1)
        sig2 = mh.compute_minhash(text2)
        assert not mh.is_duplicate("doc1", sig1)
        assert not mh.is_duplicate("doc2", sig2)


# ── Dolma Cleaner Tests ───────────────────────────────────────────

class TestDolmaCleaner:

    def test_gopher_good_text(self):
        config = DolmaConfig()
        gopher = GopherQualityFilter(config)
        text = (
            "The study of linguistics encompasses the analysis "
            "of language form, language meaning, and language in "
            "context. It examines how language is structured, how "
            "it is acquired, and how it changes over time across "
            "different communities and cultures around the world."
        ) * 3
        passed, score, failures = gopher.filter(text)
        assert passed, f"Should pass. Failures: {failures}"

    def test_gopher_repetitive_text_fails(self):
        config = DolmaConfig()
        gopher = GopherQualityFilter(config)
        text = "buy now sale discount click here buy now " * 100
        passed, score, failures = gopher.filter(text)
        assert not passed

    def test_pii_redactor_ssn(self):
        redactor = PIIRedactor()
        text = "My SSN is 123-45-6789 please keep it safe."
        assert redactor.has_pii(text)
        redacted, types = redactor.redact(text)
        assert "123-45-6789" not in redacted
        assert "ssn" in types

    def test_pii_redactor_email(self):
        redactor = PIIRedactor()
        text = "Email me at john.doe@example.com anytime."
        redacted, types = redactor.redact(text)
        assert "john.doe@example.com" not in redacted
        assert "email" in types

    def test_dolma_pipeline_jsonl(self, tmp_path):
        from data.cleaning.dolma_cleaner import DolmaTextCleaner
        config = DolmaConfig(
            min_length=50,
            min_words=10,
            output_dir=str(tmp_path)
        )
        cleaner = DolmaTextCleaner(config)

        # Create test input
        input_file = str(tmp_path / "test_input.jsonl")
        output_file = str(tmp_path / "test_output.jsonl")

        good_doc = {
            "url": "https://example.com/article",
            "text": (
                "This is a well-written article about machine learning "
                "and artificial intelligence. It contains many useful "
                "facts and explanations about how neural networks work, "
                "including details about training procedures and "
                "optimization algorithms used in modern deep learning."
            ),
        }
        short_doc = {
            "url": "https://example.com/short",
            "text": "Too short."
        }

        with open(input_file, 'w') as f:
            f.write(json.dumps(good_doc) + '\n')
            f.write(json.dumps(short_doc) + '\n')

        stats = cleaner.process_file(input_file, output_file)
        assert stats['total'] == 2
        assert stats['passed'] >= 1


# ── FineWeb Tests ─────────────────────────────────────────────────

class TestFineWebCleaner:

    def test_educational_scorer_high(self):
        scorer = EducationalScorer()
        text = (
            "Therefore, the hypothesis can be confirmed through "
            "analysis of the evidence. The research demonstrates "
            "that the concept is well-defined. Furthermore, "
            "the theoretical framework explains the underlying "
            "principles. The conclusion follows logically from "
            "the premises established earlier in this study."
        )
        score = scorer.score(text)
        assert score > 0.0

    def test_educational_scorer_low(self):
        scorer = EducationalScorer()
        text = (
            "Click here to subscribe! Follow us on social media. "
            "Like and share this post. Buy now and get a discount! "
            "Cookie policy. Privacy policy. Terms of service. "
            "Follow follow follow like like subscribe subscribe."
        )
        score = scorer.score(text)
        assert score < 0.5