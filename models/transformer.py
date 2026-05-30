"""
Transformer block and full model assembly.
Implements Pre-LN Transformer (used by GPT-2, LLaMA, etc.)
with optional Post-LN for BERT-style models.
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from models.layers import ModelConfig, build_norm, build_mlp
from models.attention import KVCache, build_attention


class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm Transformer Block.
    
    Architecture (Pre-LN, more stable training):
        x -> LayerNorm -> Attention -> x + residual
        x -> LayerNorm -> FFN       -> x + residual
    
    Note: Original Transformer used Post-LN.
    Pre-LN is now standard for large LMs.
    """

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        # Normalization
        self.ln_1 = build_norm(config, config.hidden_size)
        self.ln_2 = build_norm(config, config.hidden_size)

        # Attention
        self.attn = build_attention(config, layer_idx)

        # Feed-Forward Network
        self.mlp = build_mlp(config)

        # Optional layer-scale (improves deep network stability)
        self.use_layer_scale = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        output_attentions: bool = False,
        use_causal_mask: bool = True
    ) -> tuple:
        # Self-Attention with residual
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
            output_attentions=output_attentions,
            use_causal_mask=use_causal_mask
        )
        hidden_states = attn_outputs[0]
        hidden_states = residual + hidden_states

        # FFN with residual
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += attn_outputs[1:]
        return outputs


class TransformerModel(nn.Module):
    """
    Core Transformer stack (encoder or decoder).
    No embedding or LM head — those are in GPTModel.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=i)
            for i in range(config.num_layers)
        ])
        self.norm = build_norm(config, config.hidden_size)
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        use_causal_mask: bool = True
    ) -> dict:
        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpoint_layer(
                    layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,   # No KV cache with grad checkpointing
                    output_attentions,
                    use_causal_mask
                )
                layer_outputs = (hidden_states,)
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    kv_cache=kv_cache,
                    output_attentions=output_attentions,
                    use_causal_mask=use_causal_mask
                )
                hidden_states = layer_outputs[0]

            if output_attentions and len(layer_outputs) > 1:
                all_attentions.append(layer_outputs[1])

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        return {
            'last_hidden_state': hidden_states,
            'hidden_states': all_hidden_states,
            'attentions': all_attentions,
        }

    def _gradient_checkpoint_layer(self, layer, *args):
        import torch.utils.checkpoint as checkpoint
        def custom_forward(*inputs):
            return layer(*inputs)[0]
        return checkpoint.checkpoint(custom_forward, *args)