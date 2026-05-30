"""
Unit tests for model components.
"""

import pytest
import torch

from models.layers import ModelConfig, RMSNorm, RotaryEmbedding, SwiGLU
from models.attention import MultiHeadAttention, GroupedQueryAttention
from models.transformer import TransformerBlock
from models.gpt import GPTModel


@pytest.fixture
def tiny_config():
    return ModelConfig(
        vocab_size=1000,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=128,
        max_position_embeddings=128,
        dropout=0.0,
        use_rope=True,
        use_gqa=False,
        use_swiglu=True,
        use_rmsnorm=True,
        tie_word_embeddings=True,
    )


def test_rmsnorm(tiny_config):
    norm = RMSNorm(tiny_config.hidden_size)
    x = torch.randn(2, 10, tiny_config.hidden_size)
    out = norm(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()


def test_rotary_embedding(tiny_config):
    rope = RotaryEmbedding(
        tiny_config.hidden_size // tiny_config.num_heads
    )
    B, H, S = 2, tiny_config.num_heads, 10
    D = tiny_config.hidden_size // H
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    q_rot, k_rot = rope(q, k)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


def test_swiglu(tiny_config):
    ffn = SwiGLU(
        tiny_config.hidden_size,
        tiny_config.intermediate_size
    )
    x = torch.randn(2, 10, tiny_config.hidden_size)
    out = ffn(x)
    assert out.shape == x.shape


def test_mha(tiny_config):
    attn = MultiHeadAttention(tiny_config, layer_idx=0)
    x = torch.randn(2, 10, tiny_config.hidden_size)
    out = attn(x)
    assert out[0].shape == x.shape


def test_gqa(tiny_config):
    tiny_config.use_gqa = True
    attn = GroupedQueryAttention(tiny_config, layer_idx=0)
    x = torch.randn(2, 10, tiny_config.hidden_size)
    out = attn(x)
    assert out[0].shape == x.shape


def test_transformer_block(tiny_config):
    block = TransformerBlock(tiny_config, layer_idx=0)
    x = torch.randn(2, 10, tiny_config.hidden_size)
    out = block(x)
    assert out[0].shape == x.shape


def test_gpt_forward(tiny_config):
    model = GPTModel(tiny_config)
    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 10))
    labels = torch.randint(0, tiny_config.vocab_size, (2, 10))
    out = model(input_ids=input_ids, labels=labels)
    assert out.logits.shape == (2, 10, tiny_config.vocab_size)
    assert out.loss is not None
    assert out.loss.item() > 0


def test_gpt_generation(tiny_config):
    from generation.strategies import GenerationConfig, GreedyDecoder
    model = GPTModel(tiny_config)
    input_ids = torch.randint(0, tiny_config.vocab_size, (1, 5))
    config = GenerationConfig(
        max_new_tokens=10,
        eos_token_id=tiny_config.eos_token_id
    )
    decoder = GreedyDecoder(config)
    output = decoder.generate(model, input_ids)
    assert output.shape[1] > input_ids.shape[1]


def test_weight_tying(tiny_config):
    model = GPTModel(tiny_config)
    assert model.lm_head.weight is (
        model.embeddings.token_embedding.weight
    )


def test_perplexity(tiny_config):
    model = GPTModel(tiny_config)
    input_ids = torch.randint(0, tiny_config.vocab_size, (1, 20))
    ppl = model.compute_perplexity(input_ids)
    assert ppl > 0
    assert not torch.isnan(torch.tensor(ppl))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])