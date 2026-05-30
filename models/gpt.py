"""
GPT-style causal language model.
Full implementation with:
- Token + Position embeddings
- Transformer decoder stack
- LM Head with weight tying
- Pretraining and inference modes
- Perplexity computation
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import ModelConfig, build_norm
from models.transformer import TransformerModel
from models.attention import KVCache


@dataclass
class GPTOutput:
    loss: Optional[torch.Tensor] = None
    logits: torch.Tensor = None
    hidden_states: Optional[list] = None
    attentions: Optional[list] = None
    past_key_values: Optional[KVCache] = None


class GPTEmbeddings(nn.Module):
    """
    Token + Position embeddings.
    
    GPT-2 uses learned absolute position embeddings.
    If use_rope=True, only token embeddings are used
    (position info handled in attention via RoPE).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id
        )

        # Learned absolute positions (GPT-2 style)
        # Not used when RoPE is enabled
        if not config.use_rope:
            self.position_embedding = nn.Embedding(
                config.max_position_embeddings,
                config.hidden_size
            )

        self.dropout = nn.Dropout(config.dropout)
        self.config = config

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, S = input_ids.shape
        token_emb = self.token_embedding(input_ids)

        if not self.config.use_rope:
            if position_ids is None:
                position_ids = torch.arange(
                    S, device=input_ids.device
                ).unsqueeze(0)
            pos_emb = self.position_embedding(position_ids)
            token_emb = token_emb + pos_emb

        return self.dropout(token_emb)


class GPTModel(nn.Module):
    """
    Full GPT Causal Language Model.
    
    Architecture:
        Input IDs
            -> Embeddings (token + optional position)
            -> N x TransformerBlock (Pre-LN)
            -> Final LayerNorm
            -> LM Head (linear, optionally tied to embeddings)
    
    Supports families:
        - GPT-2 (absolute pos, LayerNorm, GeLU)
        - GPT-NeoX / LLaMA style (RoPE, RMSNorm, SwiGLU)
        - DeepSeek / Qwen style (GQA, RoPE, SwiGLU)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embeddings = GPTEmbeddings(config)
        self.transformer = TransformerModel(config)

        # LM Head: hidden_size -> vocab_size
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False
        )

        # Weight tying: share token embedding and LM head weights
        if config.tie_word_embeddings:
            self.lm_head.weight = (
                self.embeddings.token_embedding.weight
            )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following GPT-2 paper."""
        std = self.config.initializer_range

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

        # Scale residual projections by 1/sqrt(num_layers)
        # (GPT-2 style initialization)
        for name, param in self.named_parameters():
            if name.endswith(('out_proj.weight', 'down_proj.weight')):
                nn.init.normal_(
                    param,
                    mean=0.0,
                    std=std / math.sqrt(2 * self.config.num_layers)
                )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings.token_embedding

    def set_input_embeddings(self, embeddings: nn.Embedding):
        self.embeddings.token_embedding = embeddings

    def enable_gradient_checkpointing(self):
        self.transformer.gradient_checkpointing = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        use_causal_mask: bool = True
    ) -> GPTOutput:
        B, S = input_ids.shape

        # Position IDs
        if position_ids is None:
            past_len = (
                kv_cache.get_seq_length() if kv_cache else 0
            )
            position_ids = torch.arange(
                past_len, past_len + S,
                device=input_ids.device
            ).unsqueeze(0)

        # Embeddings
        hidden_states = self.embeddings(input_ids, position_ids)

        # Process attention mask to additive format
        extended_mask = None
        if attention_mask is not None:
            extended_mask = self._prepare_attention_mask(
                attention_mask, hidden_states.dtype
            )

        # Transformer
        transformer_out = self.transformer(
            hidden_states,
            attention_mask=extended_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_causal_mask=use_causal_mask
        )

        hidden_states = transformer_out['last_hidden_state']

        # LM Head
        logits = self.lm_head(hidden_states)

        # Loss computation
        loss = None
        if labels is not None:
            loss = self._compute_loss(logits, labels)

        return GPTOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_out.get('hidden_states'),
            attentions=transformer_out.get('attentions'),
            past_key_values=kv_cache
        )

    def _prepare_attention_mask(
        self,
        attention_mask: torch.Tensor,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Convert binary attention mask (B, S) to
        additive mask (B, 1, 1, S) compatible with attention weights.
        """
        # 1 -> 0 (attend), 0 -> -inf (don't attend)
        mask = attention_mask[:, None, None, :]
        mask = (1.0 - mask.to(dtype)) * torch.finfo(dtype).min
        return mask

    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute causal language modeling loss.
        Shift logits and labels for next-token prediction.
        """
        # Shift: predict token[i+1] from token[i]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, self.config.vocab_size),
            shift_labels.view(-1),
            ignore_index=self.config.pad_token_id
        )
        return loss

    def compute_perplexity(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        stride: int = 512
    ) -> float:
        """
        Compute perplexity using sliding window for long sequences.
        Handles sequences longer than max_position_embeddings.
        """
        self.eval()
        max_length = self.config.max_position_embeddings
        seq_len = input_ids.shape[1]

        nlls = []
        prev_end = 0

        with torch.no_grad():
            for begin_loc in range(0, seq_len, stride):
                end_loc = min(begin_loc + max_length, seq_len)
                trg_len = end_loc - prev_end
                input_chunk = input_ids[:, begin_loc:end_loc]
                target_chunk = input_chunk.clone()
                target_chunk[:, :-trg_len] = (
                    self.config.pad_token_id
                )

                output = self(
                    input_chunk,
                    labels=target_chunk
                )
                nlls.append(output.loss * trg_len)
                prev_end = end_loc

                if end_loc == seq_len:
                    break

        total_loss = torch.stack(nlls).sum()
        perplexity = torch.exp(total_loss / seq_len).item()
        return perplexity

    @classmethod
    def from_config(cls, config: ModelConfig) -> 'GPTModel':
        return cls(config)

    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(
                p.numel() for p in self.parameters()
                if p.requires_grad
            )
        return sum(p.numel() for p in self.parameters())

    def save_pretrained(self, save_dir: str):
        import os, json
        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            self.state_dict(),
            os.path.join(save_dir, 'model.pt')
        )
        config_dict = vars(self.config)
        with open(
            os.path.join(save_dir, 'config.json'), 'w'
        ) as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def from_pretrained(
        cls, load_dir: str, map_location: str = 'cpu'
    ) -> 'GPTModel':
        import json
        with open(f'{load_dir}/config.json') as f:
            config_dict = json.load(f)
        config = ModelConfig(**config_dict)
        model = cls(config)
        state_dict = torch.load(
            f'{load_dir}/model.pt',
            map_location=map_location
        )
        model.load_state_dict(state_dict)
        return model