"""
Multi-Head Attention implementations:
- Standard MHA (GPT-2)
- Grouped Query Attention / GQA (LLaMA-2, Gemma)
- Flash Attention integration
- KV-Cache for efficient inference
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import ModelConfig, RotaryEmbedding


class KVCache:
    """
    Key-Value cache for autoregressive generation.
    Stores past K, V tensors to avoid recomputation.
    """

    def __init__(self):
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx >= len(self.key_cache):
            self.key_cache.append(key)
            self.value_cache.append(value)
        else:
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key], dim=2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value], dim=2
            )
        return (
            self.key_cache[layer_idx],
            self.value_cache[layer_idx]
        )

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx < len(self.key_cache):
            return self.key_cache[layer_idx].shape[2]
        return 0

    def clear(self):
        self.key_cache.clear()
        self.value_cache.clear()


def causal_mask(
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype
) -> torch.Tensor:
    """
    Create causal (lower-triangular) attention mask.
    Returns additive mask with -inf for masked positions.
    """
    mask = torch.full(
        (seq_len, seq_len), float('-inf'),
        device=device, dtype=dtype
    )
    return torch.triu(mask, diagonal=1)


class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Self-Attention (GPT-2 / BERT style).
    
    Supports:
    - Causal masking for decoder
    - Flash Attention (if available)
    - KV-Cache for inference
    - Dropout
    """

    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = self.head_dim ** -0.5

        assert config.hidden_size % config.num_heads == 0, (
            f"hidden_size {config.hidden_size} must be "
            f"divisible by num_heads {config.num_heads}"
        )

        # Fused QKV projection (more efficient)
        self.qkv_proj = nn.Linear(
            config.hidden_size,
            3 * config.hidden_size,
            bias=False
        )
        self.out_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=False
        )
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Check for Flash Attention
        self.use_flash = self._check_flash_attn()

        # RoPE
        if config.use_rope:
            self.rotary_emb = RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                theta=config.rope_theta
            )

    def _check_flash_attn(self) -> bool:
        try:
            from flash_attn import flash_attn_func
            return True
        except ImportError:
            return False

    def _split_heads(
        self, x: torch.Tensor, num_heads: int
    ) -> torch.Tensor:
        """(B, S, H) -> (B, num_heads, S, head_dim)"""
        B, S, _ = x.shape
        x = x.view(B, S, num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, num_heads, S, head_dim) -> (B, S, H)"""
        B, _, S, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(B, S, self.hidden_size)

    def _flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = True
    ) -> torch.Tensor:
        from flash_attn import flash_attn_func
        # Flash Attention expects (B, S, H, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = flash_attn_func(
            q, k, v,
            dropout_p=self.config.attention_dropout if self.training else 0.0,
            causal=causal,
            softmax_scale=self.scale
        )
        return out.transpose(1, 2)

    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Scaled dot-product attention
        # q, k, v: (B, num_heads, S, head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        attn_weights = attn_weights * self.scale

        if causal:
            seq_len = q.shape[-2]
            mask = causal_mask(
                seq_len,
                device=q.device,
                dtype=q.dtype
            )
            attn_weights = attn_weights + mask

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(q.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        return out, attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        output_attentions: bool = False,
        use_causal_mask: bool = True
    ) -> tuple:
        B, S, _ = hidden_states.shape

        # Fused QKV
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)

        # Split heads
        q = self._split_heads(q, self.num_heads)
        k = self._split_heads(k, self.num_heads)
        v = self._split_heads(v, self.num_heads)

        # Apply RoPE
        if self.config.use_rope:
            q, k = self.rotary_emb(q, k, position_ids)

        # KV Cache update
        if kv_cache is not None:
            k, v = kv_cache.update(self.layer_idx, k, v)

        # Attention
        if self.use_flash and not output_attentions:
            attn_out = self._flash_attention(
                q, k, v, causal=use_causal_mask
            )
            attn_weights = None
        else:
            attn_out, attn_weights = self._standard_attention(
                q, k, v, attention_mask, use_causal_mask
            )

        # Merge heads and project
        attn_out = self._merge_heads(attn_out)
        attn_out = self.out_proj(attn_out)
        attn_out = self.resid_dropout(attn_out)

        outputs = (attn_out,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA).
    
    Key insight: num_kv_heads < num_heads.
    Multiple query heads share one K/V head.
    Reduces memory bandwidth for K/V cache by num_heads/num_kv_heads.
    
    Special cases:
    - GQA with num_kv_heads=1: Multi-Query Attention (MQA)
    - GQA with num_kv_heads=num_heads: Standard MHA
    
    Used in: LLaMA-2 (70B), Mistral, Qwen, DeepSeek.
    """

    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = self.head_dim ** -0.5
        self.groups = config.num_heads // config.num_kv_heads

        assert config.num_heads % config.num_kv_heads == 0, (
            "num_heads must be divisible by num_kv_heads"
        )

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_heads * self.head_dim,
            bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_kv_heads * self.head_dim,
            bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_kv_heads * self.head_dim,
            bias=False
        )
        self.out_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=False
        )
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        if config.use_rope:
            self.rotary_emb = RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                theta=config.rope_theta
            )

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expand KV heads to match query heads.
        (B, num_kv_heads, S, D) -> (B, num_heads, S, D)
        """
        if self.groups == 1:
            return x
        B, num_kv, S, D = x.shape
        x = x.unsqueeze(2).expand(B, num_kv, self.groups, S, D)
        return x.reshape(B, num_kv * self.groups, S, D)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        output_attentions: bool = False,
        use_causal_mask: bool = True
    ) -> tuple:
        B, S, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.config.use_rope:
            q, k = self.rotary_emb(q, k, position_ids)

        if kv_cache is not None:
            k, v = kv_cache.update(self.layer_idx, k, v)

        # Repeat KV heads for grouped query
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if use_causal_mask:
            mask = causal_mask(S, q.device, q.dtype)
            attn_weights = attn_weights + mask

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(q.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, S, self.hidden_size)
        out = self.out_proj(out)
        out = self.resid_dropout(out)

        outputs = (out,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


def build_attention(
    config: ModelConfig, layer_idx: int
) -> nn.Module:
    """Attention factory: MHA or GQA."""
    if config.use_gqa:
        return GroupedQueryAttention(config, layer_idx)
    return MultiHeadAttention(config, layer_idx)