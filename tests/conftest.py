"""
pytest configuration and shared fixtures.

Fixtures defined here are available to ALL test files without imports.
This avoids duplicating setup code across test files.
"""

import pytest
import torch


@pytest.fixture
def device():
    """Get the available device (cuda if available, else cpu)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def small_config():
    """Small model config for fast tests (runs on CPU in <1 second)."""
    from models.layers import ModelConfig

    return ModelConfig(
        vocab_size=1000,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        num_kv_heads=4,
        intermediate_size=512,
        max_position_embeddings=256,
        use_rope=True,
        use_rmsnorm=True,
        use_swiglu=True,
        use_gqa=False,
        tie_word_embeddings=True,
    )


@pytest.fixture
def small_model(small_config):
    """A small GPT model for testing (no pretrained weights)."""
    from models.gpt import GPTModel

    return GPTModel(small_config)


@pytest.fixture
def mock_tokenizer():
    """A simple mock tokenizer for testing generation without training a real one."""

    class MockTokenizer:
        def __init__(self):
            self.vocab_size = 1000

        def encode(self, text: str) -> list[int]:
            # Simple: convert each char to its ordinal, mod vocab_size
            return [ord(c) % self.vocab_size for c in text]

        def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
            return "".join(chr(i) for i in ids)

        def encode_batch(self, texts, padding=True, max_length=128, truncation=True):
            encoded = [self.encode(t)[:max_length] for t in texts]
            if padding:
                max_len = max(len(e) for e in encoded)
                encoded = [e + [0] * (max_len - len(e)) for e in encoded]
                masks = [[1] * len(e) + [0] * (max_len - len(e)) for e in encoded]
            else:
                masks = [[1] * len(e) for e in encoded]
            return {"input_ids": encoded, "attention_mask": masks}

    return MockTokenizer()


@pytest.fixture
def sample_input_ids():
    """Sample input token IDs for generation tests."""
    return torch.randint(0, 1000, (1, 16))


@pytest.fixture
def temp_checkpoint(tmp_path, small_model):
    """Save and return path to a temporary model checkpoint."""
    save_dir = tmp_path / "test_model"
    save_dir.mkdir()
    small_model.save_pretrained(str(save_dir))
    return str(save_dir)


@pytest.fixture(scope="session")
def test_settings():
    """Override settings for testing (DEV_MODE=true, no real secrets)."""
    import os
    os.environ["DEV_MODE"] = "true"
    os.environ["JWT_SECRET_KEY"] = "test-secret-key-for-testing-only-32chars"
    os.environ["ENVIRONMENT"] = "development"
    from config.settings import Settings
    return Settings()
