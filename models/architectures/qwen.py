"""
QwenModel — full causal LM wrapper.
Appended to qwen.py.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from models.layers import RMSNorm, SwiGLU
from models.attention import KVCache
from models.gpt import GPTOutput


class QwenBlock(nn.Module):
    """Single Qwen transformer block."""

    def __init__(self, config: "QwenConfig", layer_idx: int):
        super().__init__()
        self.ln_1 = RMSNorm(config.hidden_size, config.layer_norm_eps)
        self.ln_2 = RMSNorm(config.hidden_size, config.layer_norm_eps)
        self.attn = QwenAttention(config, layer_idx)
        self.mlp = SwiGLU(
            config.hidden_size, config.intermediate_size
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        output_attentions: bool = False,
        use_causal_mask: bool = True,
    ) -> tuple:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_out = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
            output_attentions=output_attentions,
            use_causal_mask=use_causal_mask,
        )
        hidden_states = residual + attn_out[0]

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += attn_out[1:]
        return outputs


class QwenModel(nn.Module):
    """
    Full Qwen causal language model.
    Features: Dynamic NTK RoPE, Log-N attention,
    SwiGLU FFN, RMSNorm, no bias.
    """

    def __init__(self, config: "QwenConfig"):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList([
            QwenBlock(config, layer_idx=i)
            for i in range(config.num_layers)
        ])
        self.norm = RMSNorm(
            config.hidden_size, config.layer_norm_eps
        )
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        std = self.config.initializer_range
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        use_causal_mask: bool = True,
    ) -> GPTOutput:
        B, S = input_ids.shape

        if position_ids is None:
            past_len = kv_cache.get_seq_length() if kv_cache else 0
            position_ids = torch.arange(
                past_len, past_len + S, device=input_ids.device
            ).unsqueeze(0)

        x = self.embedding(input_ids)
        all_hidden = [] if output_hidden_states else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden.append(x)

            x = layer(
                x,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
                output_attentions=output_attentions,
                use_causal_mask=use_causal_mask,
            )[0]

        x = self.norm(x)
        if output_hidden_states:
            all_hidden.append(x)

        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id,
            )

        return GPTOutput(
            loss=loss,
            logits=logits,
            hidden_states=all_hidden,
        )

    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(
                p.numel() for p in self.parameters()
                if p.requires_grad
            )
        return sum(p.numel() for p in self.parameters())