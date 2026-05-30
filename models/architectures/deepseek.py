"""
DeepSeekModel — full causal LM wrapper
that accumulates MoE aux losses correctly.
Appended to deepseek.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass

from models.layers import RMSNorm
from models.attention import KVCache
from models.gpt import GPTOutput


class DeepSeekModel(nn.Module):
    """
    Full DeepSeek causal language model.
    Wraps DeepSeekBlock layers and accumulates
    MoE auxiliary losses for load balancing.
    """

    def __init__(self, config: "DeepSeekConfig"):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList([
            DeepSeekBlock(config, layer_idx=i)
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
        total_aux_loss = torch.tensor(0.0, device=input_ids.device)
        all_hidden = [] if output_hidden_states else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden.append(x)

            layer_out = layer(
                x,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
                output_attentions=output_attentions,
                use_causal_mask=use_causal_mask,
            )
            x = layer_out[0]

            # Accumulate MoE auxiliary loss
            for item in layer_out[1:]:
                if (
                    isinstance(item, torch.Tensor)
                    and item.ndim == 0
                ):
                    total_aux_loss = total_aux_loss + item

        x = self.norm(x)
        if output_hidden_states:
            all_hidden.append(x)

        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id,
            )
            # Add MoE auxiliary loss
            loss = ce_loss + total_aux_loss

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