"""Tests for text generation strategies."""

import pytest
import torch

from generation.strategies import (
    GenerationConfig,
    GreedyDecoder,
    SamplingDecoder,
    BeamSearchDecoder,
    top_k_filtering,
    top_p_filtering,
    TemperatureScaling,
    RepetitionPenalty,
)
from models.layers import ModelConfig
from models.gpt import GPTModel


@pytest.fixture
def tiny_model():
    config = ModelConfig(
        vocab_size=500,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
        dropout=0.0,
    )
    return GPTModel(config)


@pytest.fixture
def gen_config():
    return GenerationConfig(
        max_new_tokens=10,
        temperature=1.0,
        top_k=50,
        top_p=0.9,
        eos_token_id=2,
        pad_token_id=0,
    )


def test_top_k_filtering():
    logits = torch.randn(2, 100)
    filtered = top_k_filtering(logits, top_k=10)
    # Each row should have exactly 10 non-inf values
    for row in filtered:
        finite = (row != float("-inf")).sum().item()
        assert finite == 10


def test_top_p_filtering():
    logits = torch.randn(2, 100)
    filtered = top_p_filtering(logits, top_p=0.9)
    assert filtered.shape == logits.shape


def test_temperature_scaling():
    proc = TemperatureScaling(0.5)
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    input_ids = torch.zeros(1, 5, dtype=torch.long)
    out = proc(input_ids, logits)
    expected = logits / 0.5
    assert torch.allclose(out, expected)


def test_repetition_penalty():
    proc = RepetitionPenalty(1.5)
    input_ids = torch.tensor([[1, 2, 3]])
    logits = torch.ones(1, 10)
    out = proc(input_ids, logits)
    # Tokens 1, 2, 3 should be penalized (positive -> divided)
    assert out[0, 1] < logits[0, 1]
    assert out[0, 2] < logits[0, 2]


def test_greedy_decoder(tiny_model, gen_config):
    decoder = GreedyDecoder(gen_config)
    input_ids = torch.randint(10, 490, (1, 5))
    output = decoder.generate(tiny_model, input_ids)
    assert output.shape[0] == 1
    assert output.shape[1] >= input_ids.shape[1]


def test_sampling_decoder(tiny_model, gen_config):
    decoder = SamplingDecoder(gen_config)
    input_ids = torch.randint(10, 490, (1, 5))
    output = decoder.generate(tiny_model, input_ids)
    assert output.shape[0] == 1
    assert output.shape[1] >= input_ids.shape[1]


def test_sampling_stream(tiny_model, gen_config):
    decoder = SamplingDecoder(gen_config)
    input_ids = torch.randint(10, 490, (1, 5))
    tokens = list(decoder.generate_stream(tiny_model, input_ids))
    assert len(tokens) > 0
    assert all(t.shape == (1, 1) for t in tokens)


def test_beam_search(tiny_model):
    config = GenerationConfig(
        max_new_tokens=5,
        num_beams=2,
        eos_token_id=2,
        pad_token_id=0,
    )
    decoder = BeamSearchDecoder(config)
    input_ids = torch.randint(10, 490, (1, 5))
    output = decoder.generate(tiny_model, input_ids)
    assert output.shape[0] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])