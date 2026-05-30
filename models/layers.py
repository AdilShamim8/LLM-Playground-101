"""
Core neural network layers for Transformer architecture.
Implements RMSNorm, RoPE, GQA, SwiGLU, and other modern components.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    num_kv_heads: int = 12          # For GQA (set < num_heads)
    intermediate_size: int = 3072
    max_position_embeddings: int = 2048
    dropout: float = 0.1
    attention_dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    use_rope: bool = True
    use_gqa: bool = False
    use_swiglu: bool = True
    use_rmsnorm: bool = True
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    architecture: str = "gpt"       # gpt | gemma | deepseek


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Used in LLaMA, Gemma, DeepSeek instead of LayerNorm.
    Faster: no mean subtraction.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, hidden_size)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_norm


class LayerNorm(nn.Module):
    """Standard LayerNorm with optional bias (GPT-2 style)."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        bias: bool = True
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(
            torch.zeros(hidden_size)
        ) if bias else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x, x.shape[-1:], self.weight, self.bias, self.eps
        )


def build_norm(config: ModelConfig, hidden_size: int) -> nn.Module:
    """Factory for normalization layer."""
    if config.use_rmsnorm:
        return RMSNorm(hidden_size, eps=config.layer_norm_eps)
    return LayerNorm(hidden_size, eps=config.layer_norm_eps)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    
    Encodes position information directly into Q, K vectors
    via rotation. Key property: relative positions encoded
    via dot product — no absolute position tokens needed.
    
    Used in: GPT-NeoX, LLaMA, Gemma, DeepSeek, Qwen.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        theta: float = 10000.0,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.theta = theta

        # Compute inverse frequencies: shape (dim/2,)
        inv_freq = 1.0 / (
            theta ** (
                torch.arange(0, dim, 2, dtype=torch.float32, device=device)
                / dim
            )
        )
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self._build_cache(max_position_embeddings)

    def _build_cache(self, seq_len: int):
        t = torch.arange(
            seq_len,
            device=self.inv_freq.device,
            dtype=self.inv_freq.dtype
        )
        # Outer product: (seq_len, dim/2)
        freqs = torch.outer(t, self.inv_freq)
        # Concatenate to get (seq_len, dim)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[-2]

        if seq_len > self.max_position_embeddings:
            self._build_cache(seq_len)

        if position_ids is not None:
            cos = self.cos_cached[position_ids]
            sin = self.sin_cached[position_ids]
        else:
            cos = self.cos_cached[:seq_len]
            sin = self.sin_cached[:seq_len]

        q_rot = apply_rotary_emb(q, cos, sin)
        k_rot = apply_rotary_emb(k, cos, sin)
        return q_rot, k_rot


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the latter half of the hidden dims of x."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> torch.Tensor:
    """Apply RoPE to query or key tensor."""
    # x: (..., seq_len, dim)
    # cos/sin: (seq_len, dim)
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin


class SwiGLU(nn.Module):
    """
    SwiGLU activation function.
    
    SwiGLU(x) = Swish(xW + b) ⊙ (xV + c)
    
    Used in: PaLM, LLaMA, Gemma, DeepSeek.
    Outperforms ReLU/GELU in practice.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False
    ):
        super().__init__()
        self.gate_proj = nn.Linear(
            hidden_size, intermediate_size, bias=bias
        )
        self.up_proj = nn.Linear(
            hidden_size, intermediate_size, bias=bias
        )
        self.down_proj = nn.Linear(
            intermediate_size, hidden_size, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class GPT2MLP(nn.Module):
    """Standard GPT-2 MLP with GELU activation."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)


def build_mlp(config: ModelConfig) -> nn.Module:
    """Factory for MLP block."""
    if config.use_swiglu:
        return SwiGLU(
            config.hidden_size,
            config.intermediate_size
        )
    return GPT2MLP(
        config.hidden_size,
        config.intermediate_size,
        config.dropout
    )