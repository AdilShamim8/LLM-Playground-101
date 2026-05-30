"""Tests for training pipeline components."""

import os
import tempfile
import pytest
import torch
import torch.nn as nn

from models.layers import ModelConfig
from models.gpt import GPTModel
from training.pretrain import (
    PretrainConfig,
    PretrainDataset,
    CosineSchedulerWithWarmup,
    PretrainTrainer,
)
from training.sft import SFTConfig, ChatFormatter, SFTDataset
from training.rlhf.reward_model import (
    RewardModelConfig, RewardModel, PreferenceDataset
)
from training.rlhf.verifiable_tasks import (
    MathVerifier, CodeVerifier, FormatVerifier
)
from evaluation.metrics import BLEUScore, ROUGEScore
from evaluation.human_eval import ELORatingSystem, ComparisonResult
import time


@pytest.fixture
def tiny_config():
    return ModelConfig(
        vocab_size=500,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
        dropout=0.0,
    )


@pytest.fixture
def tiny_model(tiny_config):
    return GPTModel(tiny_config)


# ── Pretrain Tests ────────────────────────────────────────────────

class TestPretraining:

    def test_cosine_scheduler_warmup(self):
        model = nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-3
        )
        scheduler = CosineSchedulerWithWarmup(
            optimizer,
            warmup_steps=10,
            max_steps=100,
            min_lr=1e-5,
            max_lr=1e-3
        )
        # During warmup: LR should increase
        lrs = []
        for _ in range(10):
            scheduler.step()
            lrs.append(scheduler.get_lr())
        assert lrs[-1] > lrs[0]

        # After warmup: LR should decrease (cosine)
        mid_lr = scheduler.get_lr()
        for _ in range(50):
            scheduler.step()
        end_lr = scheduler.get_lr()
        assert end_lr < mid_lr

    def test_scheduler_min_lr(self):
        model = nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-3
        )
        scheduler = CosineSchedulerWithWarmup(
            optimizer,
            warmup_steps=5,
            max_steps=100,
            min_lr=1e-5,
            max_lr=1e-3
        )
        for _ in range(200):
            scheduler.step()
        assert scheduler.get_lr() >= 1e-5

    def test_pretrain_dataset_tokens(self, tmp_path):
        # Create a fake binary token file
        import numpy as np
        tokens = np.random.randint(0, 500, size=1000, dtype=np.uint16)
        fpath = str(tmp_path / "train_000.bin")
        tokens.tofile(fpath)

        dataset = PretrainDataset(
            data_path=str(tmp_path),
            seq_length=32,
            split="train"
        )
        assert len(dataset) > 0
        item = dataset[0]
        assert "input_ids" in item
        assert "labels" in item
        assert item["input_ids"].shape[0] == 32
        assert item["labels"].shape[0] == 32

    def test_pretrain_forward_pass(self, tiny_model):
        input_ids = torch.randint(0, 500, (2, 16))
        labels = torch.randint(0, 500, (2, 16))
        out = tiny_model(input_ids=input_ids, labels=labels)
        assert out.loss is not None
        assert out.loss.item() > 0
        assert not torch.isnan(out.loss)

    def test_pretrain_backward(self, tiny_model):
        optimizer = torch.optim.AdamW(
            tiny_model.parameters(), lr=1e-4
        )
        input_ids = torch.randint(0, 500, (2, 16))
        labels = torch.randint(0, 500, (2, 16))

        loss_before = tiny_model(
            input_ids=input_ids, labels=labels
        ).loss.item()

        for _ in range(5):
            out = tiny_model(input_ids=input_ids, labels=labels)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        loss_after = tiny_model(
            input_ids=input_ids, labels=labels
        ).loss.item()
        # Loss should decrease (overfit on tiny batch)
        assert loss_after < loss_before


# ── SFT Tests ─────────────────────────────────────────────────────

class TestSFT:

    def test_chat_formatter_chatml(self):
        formatter = ChatFormatter(
            template="chatml",
            system_prompt="You are helpful."
        )
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        class MockTokenizer:
            def encode(self, text, **kwargs):
                return list(range(len(text.split())))
            def decode(self, ids, **kwargs):
                return "decoded"

        result = formatter.format_conversation(
            messages, MockTokenizer(), max_length=512
        )
        assert "input_ids" in result
        assert "labels" in result
        assert "attention_mask" in result
        assert len(result["input_ids"]) == 512

    def test_loss_masking(self):
        """Labels should have -100 for non-assistant tokens."""
        formatter = ChatFormatter(template="chatml")
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "It is 4."},
        ]

        class MockTokenizer:
            def encode(self, text, **kwargs):
                return list(range(min(len(text.split()), 100)))
            def decode(self, ids, **kwargs):
                return "decoded"

        result = formatter.format_conversation(
            messages, MockTokenizer(), max_length=128
        )
        labels = result["labels"]
        # Some labels should be -100 (user/system tokens)
        has_masked = any(l == -100 for l in labels)
        assert has_masked


# ── Reward Model Tests ────────────────────────────────────────────

class TestRewardModel:

    def test_reward_model_forward(self, tiny_model, tiny_config):
        rm_config = RewardModelConfig(
            hidden_size=tiny_config.hidden_size
        )
        rm = RewardModel(tiny_model, rm_config)
        input_ids = torch.randint(0, 500, (2, 16))
        rewards = rm(input_ids)
        assert rewards.shape == (2,)
        assert not torch.isnan(rewards).any()

    def test_preference_loss(self, tiny_model, tiny_config):
        rm_config = RewardModelConfig(
            hidden_size=tiny_config.hidden_size
        )
        rm = RewardModel(tiny_model, rm_config)

        chosen = torch.randint(0, 500, (2, 16))
        rejected = torch.randint(0, 500, (2, 16))

        loss, metrics = rm.compute_preference_loss(
            chosen, rejected
        )
        assert loss.item() > 0
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_preference_dataset_demo(self):
        class MockTokenizer:
            def encode(self, text, **kwargs):
                return list(range(min(50, len(text.split()))))

        dataset = PreferenceDataset(
            data_path="nonexistent_path",
            tokenizer=MockTokenizer(),
            max_length=64
        )
        assert len(dataset) > 0
        item = dataset[0]
        assert "chosen_input_ids" in item
        assert "rejected_input_ids" in item


# ── Verifiable Tasks Tests ────────────────────────────────────────

class TestVerifiableTasks:

    def test_math_verifier_correct(self):
        verifier = MathVerifier()
        result = verifier.compute_reward(
            prompt="What is 2 + 2?",
            response="The answer is 4.",
            ground_truth="4"
        )
        assert result.passed
        assert result.reward == 1.0

    def test_math_verifier_wrong(self):
        verifier = MathVerifier()
        result = verifier.compute_reward(
            prompt="What is 2 + 2?",
            response="The answer is 5.",
            ground_truth="4"
        )
        assert not result.passed
        assert result.reward < 1.0

    def test_math_extract_answer_boxed(self):
        verifier = MathVerifier()
        text = r"Therefore \boxed{42} is the answer."
        answer = verifier.extract_answer(text)
        assert answer == "42"

    def test_code_verifier_syntax_error(self):
        verifier = CodeVerifier()
        result = verifier.compute_reward(
            prompt="Write code",
            response="def foo(:\n    pass",
            ground_truth=""
        )
        assert not result.passed
        assert result.reward < 0

    def test_code_verifier_correct(self):
        verifier = CodeVerifier(timeout=5)
        result = verifier.compute_reward(
            prompt="Print hello",
            response="```python\nprint('hello')\n```",
            ground_truth="hello"
        )
        # Should execute and compare output
        assert result.reward >= 0

    def test_format_verifier_json_valid(self):
        verifier = FormatVerifier()
        result = verifier.compute_reward(
            prompt="Return JSON",
            response='{"key": "value", "number": 42}',
            ground_truth="json"
        )
        assert result.passed
        assert result.reward == 1.0

    def test_format_verifier_json_invalid(self):
        verifier = FormatVerifier()
        result = verifier.compute_reward(
            prompt="Return JSON",
            response="This is not JSON at all.",
            ground_truth="json"
        )
        assert not result.passed
        assert result.reward == 0.0


# ── Evaluation Tests ──────────────────────────────────────────────

class TestEvaluationMetrics:

    def test_bleu_perfect(self):
        bleu = BLEUScore()
        hyps = ["the cat sat on the mat"]
        refs = [["the cat sat on the mat"]]
        result = bleu.compute(hyps, refs)
        assert result["bleu"] > 90.0

    def test_bleu_empty(self):
        bleu = BLEUScore()
        result = bleu.compute([""], [[""]])
        assert result["bleu"] == 0.0

    def test_bleu_partial_match(self):
        bleu = BLEUScore()
        hyps = ["the cat sat on the mat"]
        refs = [["the dog sat on the rug"]]
        result = bleu.compute(hyps, refs)
        assert 0.0 < result["bleu"] < 100.0

    def test_rouge_perfect(self):
        rouge = ROUGEScore()
        hyps = ["the cat sat on the mat"]
        refs = ["the cat sat on the mat"]
        result = rouge.compute(hyps, refs)
        assert result["rouge1"] > 99.0
        assert result["rouge2"] > 99.0
        assert result["rougeL"] > 99.0

    def test_rouge_no_overlap(self):
        rouge = ROUGEScore()
        hyps = ["apple orange banana"]
        refs = ["car truck motorcycle"]
        result = rouge.compute(hyps, refs)
        assert result["rouge1"] == 0.0

    def test_rouge_partial(self):
        rouge = ROUGEScore()
        hyps = ["the quick brown fox"]
        refs = ["the slow brown dog"]
        result = rouge.compute(hyps, refs)
        assert result["rouge1"] > 0.0
        assert result["rouge1"] < 100.0


# ── ELO / Leaderboard Tests ───────────────────────────────────────

class TestHumanEval:

    def test_elo_initial_rating(self):
        elo = ELORatingSystem(k_factor=32)
        result = ComparisonResult(
            comparison_id="test-1",
            prompt="What is ML?",
            model_a="model_a",
            model_b="model_b",
            response_a="ML is...",
            response_b="Machine learning is...",
            winner="a",
            judge_id="user1",
            timestamp=time.time()
        )
        elo.update(result)
        assert "model_a" in elo.ratings
        assert "model_b" in elo.ratings
        assert elo.ratings["model_a"].elo > 1000.0
        assert elo.ratings["model_b"].elo < 1000.0

    def test_elo_tie(self):
        elo = ELORatingSystem()
        result = ComparisonResult(
            comparison_id="test-tie",
            prompt="Test",
            model_a="m1", model_b="m2",
            response_a="resp", response_b="resp",
            winner="tie",
            judge_id="user1",
            timestamp=time.time()
        )
        elo.update(result)
        # Tie from equal ratings: both stay near 1000
        assert abs(elo.ratings["m1"].elo - 1000.0) < 20
        assert abs(elo.ratings["m2"].elo - 1000.0) < 20

    def test_leaderboard_ordering(self):
        elo = ELORatingSystem()
        # Model A wins all
        for i in range(10):
            result = ComparisonResult(
                comparison_id=f"test-{i}",
                prompt="Test",
                model_a="strong_model",
                model_b="weak_model",
                response_a="Good response",
                response_b="Bad response",
                winner="a",
                judge_id=f"user{i}",
                timestamp=time.time()
            )
            elo.update(result)

        board = elo.get_leaderboard()
        assert board[0]["model"] == "strong_model"
        assert board[1]["model"] == "weak_model"
        assert board[0]["elo"] > board[1]["elo"]

    def test_expected_score_symmetry(self):
        elo = ELORatingSystem()
        e_a = elo.expected_score(1000, 1000)
        assert abs(e_a - 0.5) < 1e-6

        e_a_better = elo.expected_score(1200, 1000)
        assert e_a_better > 0.5

    def test_human_eval_collector(self, tmp_path):
        from evaluation.human_eval import HumanEvalCollector
        collector = HumanEvalCollector(
            storage_path=str(tmp_path)
        )
        result = collector.submit_comparison(
            prompt="What is AI?",
            model_a="gpt-4",
            response_a="AI is artificial intelligence.",
            model_b="gpt-3",
            response_b="AI stands for artificial intelligence.",
            winner="a",
            judge_id="test_user"
        )
        assert result.comparison_id is not None
        board = collector.get_leaderboard()
        assert len(board) == 2
        assert board[0]["model"] == "gpt-4"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])