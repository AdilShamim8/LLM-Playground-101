"""
Gemma-style architecture (Google, 2024).
Key differences from standard GPT:
- Logit soft-capping (tanh-based)
- Squared ReLU or GeLU activation
- Normalized input embeddings
- Per-layer RoPE
- No bias terms anywhere
- Specific initialization scheme
Reference: Gemma: Open Models Based on Gemini (2024)
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import ModelConfig, RMSNorm, RotaryEmbedding


@dataclass
class GemmaConfig(ModelConfig):
    # Gemma-specific
    logit_soft_cap: float = 30.0        # Tanh soft-capping
    attn_logit_softcap: float = 50.0    # Attention logit cap
    use_logit_softcap: bool = True
    embedding_multiplier: float = 1.0   # sqrt(hidden_size)
    hidden_act: str = "gelu"            # "gelu" or "relu2"

    def __post_init__(self):
        # Gemma uses sqrt(hidden_size) as embedding scale
        self.embedding_multiplier = math.sqrt(self.hidden_size)


class GemmaRMSNorm(nn.Module):
    """
    Gemma uses (1 + weight) instead of weight in RMSNorm.
    This allows the initial value to be 1 + 0 = 1 even
    when weights are initialized to 0.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps)
        # Key: (1 + weight) instead of weight
        return x_norm * (1.0 + self.weight)


class GemmaMLP(nn.Module):
    """
    Gemma MLP with GeGLU or Squared ReLU.
    GeGLU = GeLU(xW) * xV  (gated variant)
    """

    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.act = config.hidden_act

    def _activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.act == "relu2":
            return F.relu(x) ** 2
        return F.gelu(x, approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self._activation(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class GemmaAttention(nn.Module):
    """
    Gemma attention with:
    - Per-layer RoPE (standard)
    - Attention logit soft-capping
    - GQA support
    - No bias in any projection
    """

    def __init__(
        self, config: GemmaConfig, layer_idx: int = 0
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = self.head_dim ** -0.5
        self.groups = config.num_heads // config.num_kv_heads

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
        self.o_proj = nn.Linear(
            config.num_heads * self.head_dim,
            config.hidden_size,
            bias=False
        )

        self.rope = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            theta=config.rope_theta
        )

        self.attn_logit_softcap = (
            config.attn_logit_softcap
            if config.use_logit_softcap else None
        )

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.groups == 1:
            return x
        B, num_kv, S, D = x.shape
        x = x.unsqueeze(2).expand(
            B, num_kv, self.groups, S, D
        )
        return x.reshape(B, num_kv * self.groups, S, D)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache=None,
        output_attentions: bool = False,
        use_causal_mask: bool = True,
    ) -> tuple:
        B, S, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(
            B, S, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(hidden_states).view(
            B, S, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(hidden_states).view(
            B, S, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        q, k = self.rope(q, k, position_ids)

        if kv_cache is not None:
            k, v = kv_cache.update(self.layer_idx, k, v)

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        attn_weights = torch.matmul(
            q, k.transpose(-2, -1)
        ) * self.scale

        # Attention logit soft-capping (Gemma2 feature)
        if self.attn_logit_softcap is not None:
            attn_weights = attn_weights / self.attn_logit_softcap
            attn_weights = torch.tanh(attn_weights)
            attn_weights = attn_weights * self.attn_logit_softcap

        if use_causal_mask:
            from models.attention import causal_mask
            mask = causal_mask(S, q.device, q.dtype)
            attn_weights = attn_weights + mask

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(q.dtype)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(
            B, S, self.num_heads * self.head_dim
        )
        out = self.o_proj(out)

        return (out,) + ((attn_weights,) if output_attentions else ())


class GemmaModel(nn.Module):
    """
    Full Gemma causal language model.
    
    Key features vs GPT:
    1. GemmaRMSNorm (1 + weight formulation)
    2. Embedding scaling by sqrt(hidden_size)
    3. Logit soft-capping on final logits
    4. GQA with no bias
    5. GeGLU / Squared ReLU activation
    """

    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(
            config.vocab_size, config.hidden_size
        )

        from models.transformer import TransformerModel

        # Override layers with Gemma blocks
        self.layers = nn.ModuleList([
            self._build_gemma_block(config, i)
            for i in range(config.num_layers)
        ])
        self.norm = GemmaRMSNorm(
            config.hidden_size, config.layer_norm_eps
        )
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )

        # Weight tying
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embedding.weight

        self._init_weights()

    def _build_gemma_block(
        self, config: GemmaConfig, layer_idx: int
    ) -> nn.Module:
        """Build a single Gemma transformer block."""

        class GemmaBlock(nn.Module):
            def __init__(self, cfg, idx):
                super().__init__()
                self.ln_1 = GemmaRMSNorm(
                    cfg.hidden_size, cfg.layer_norm_eps
                )
                self.ln_2 = GemmaRMSNorm(
                    cfg.hidden_size, cfg.layer_norm_eps
                )
                self.attn = GemmaAttention(cfg, idx)
                self.mlp = GemmaMLP(cfg)

            def forward(self, x, **kwargs):
                x = x + self.attn(self.ln_1(x), **kwargs)[0]
                x = x + self.mlp(self.ln_2(x))
                return (x,)

        return GemmaBlock(config, layer_idx)

    def _init_weights(self):
        std = self.config.initializer_range
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        kv_cache=None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        use_causal_mask: bool = True,
    ):
        B, S = input_ids.shape

        # Embedding scaling (Gemma key feature)
        x = self.embedding(input_ids)
        x = x * self.config.embedding_multiplier

        if position_ids is None:
            position_ids = torch.arange(
                S, device=input_ids.device
            ).unsqueeze(0)

        # Transformer layers
        for layer in self.layers:
            x = layer(
                x,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
                output_attentions=output_attentions,
                use_causal_mask=use_causal_mask,
            )[0]

        x = self.norm(x)
        logits = self.lm_head(x)

        # Final logit soft-capping (Gemma2)
        if self.config.use_logit_softcap:
            cap = self.config.logit_soft_cap
            logits = logits / cap
            logits = torch.tanh(logits)
            logits = logits * cap

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id
            )

        from models.gpt import GPTOutput
        return GPTOutput(loss=loss, logits=logits)