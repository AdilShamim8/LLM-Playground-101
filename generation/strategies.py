"""
Text generation strategies:
- Greedy decoding
- Beam search
- Top-k sampling
- Top-p (nucleus) sampling
- Temperature scaling
- Repetition penalty
- Contrastive search
All strategies support streaming and batch generation.
"""

import math
from dataclasses import dataclass, field
from typing import Iterator, Optional

import torch
import torch.nn.functional as F
from loguru import logger


@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    min_new_tokens: int = 1
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    do_sample: bool = True
    num_beams: int = 1
    beam_search_early_stopping: bool = True
    length_penalty: float = 1.0
    eos_token_id: int = 2
    pad_token_id: int = 0
    bos_token_id: int = 1
    forced_bos_token_id: Optional[int] = None
    forced_eos_token_id: Optional[int] = None
    bad_words_ids: list = field(default_factory=list)
    num_return_sequences: int = 1


class LogitsProcessor:
    """Base class for logits processors."""

    def __call__(
        self,
        input_ids: torch.Tensor,
        logits: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError


class TemperatureScaling(LogitsProcessor):
    """
    Temperature scaling of logits.
    
    T < 1: Sharper distribution (more deterministic)
    T > 1: Flatter distribution (more random)
    T = 1: No change (standard softmax)
    """

    def __init__(self, temperature: float):
        assert temperature > 0
        self.temperature = temperature

    def __call__(
        self,
        input_ids: torch.Tensor,
        logits: torch.Tensor
    ) -> torch.Tensor:
        return logits / self.temperature


class RepetitionPenalty(LogitsProcessor):
    """
    Penalize tokens that already appear in context.
    penalty > 1 reduces probability of repeated tokens.
    """

    def __init__(self, penalty: float):
        assert penalty >= 1.0
        self.penalty = penalty

    def __call__(
        self,
        input_ids: torch.Tensor,
        logits: torch.Tensor
    ) -> torch.Tensor:
        if self.penalty == 1.0:
            return logits

        logits = logits.clone()

        # For each sequence, penalize tokens in input
        for i, seq_ids in enumerate(input_ids):
            # Get unique token ids in this sequence
            unique_ids = seq_ids.unique()
            score = logits[i, unique_ids]

            # Divide positive scores, multiply negative
            score = torch.where(
                score < 0,
                score * self.penalty,
                score / self.penalty
            )
            logits[i, unique_ids] = score

        return logits


class NoRepeatNGramLogitsProcessor(LogitsProcessor):
    """
    Prevent repeated n-grams in generation.
    Sets logit of banned tokens to -inf.
    """

    def __init__(self, ngram_size: int):
        self.ngram_size = ngram_size

    def _get_banned_tokens(
        self,
        input_ids: torch.Tensor
    ) -> list[list[int]]:
        batch_banned = []
        for seq in input_ids:
            seq_list = seq.tolist()
            banned = []
            if len(seq_list) >= self.ngram_size:
                ngrams: dict[tuple, list] = {}
                for i in range(
                    len(seq_list) - self.ngram_size + 1
                ):
                    ngram = tuple(seq_list[i:i+self.ngram_size-1])
                    next_token = seq_list[i + self.ngram_size - 1]
                    ngrams.setdefault(ngram, []).append(next_token)

                current_ngram = tuple(
                    seq_list[-(self.ngram_size - 1):]
                )
                banned = ngrams.get(current_ngram, [])
            batch_banned.append(banned)
        return batch_banned

    def __call__(
        self,
        input_ids: torch.Tensor,
        logits: torch.Tensor
    ) -> torch.Tensor:
        banned = self._get_banned_tokens(input_ids)
        for i, banned_tokens in enumerate(banned):
            for token_id in banned_tokens:
                logits[i, token_id] = float('-inf')
        return logits


class LogitsProcessorList:
    """Chain multiple logits processors."""

    def __init__(self, processors: list[LogitsProcessor]):
        self.processors = processors

    def __call__(
        self,
        input_ids: torch.Tensor,
        logits: torch.Tensor
    ) -> torch.Tensor:
        for processor in self.processors:
            logits = processor(input_ids, logits)
        return logits


def build_logits_processors(
    config: GenerationConfig
) -> LogitsProcessorList:
    processors = []

    if config.temperature != 1.0:
        processors.append(TemperatureScaling(config.temperature))

    if config.repetition_penalty != 1.0:
        processors.append(
            RepetitionPenalty(config.repetition_penalty)
        )

    if config.no_repeat_ngram_size > 0:
        processors.append(
            NoRepeatNGramLogitsProcessor(config.no_repeat_ngram_size)
        )

    return LogitsProcessorList(processors)


def top_k_filtering(
    logits: torch.Tensor, top_k: int
) -> torch.Tensor:
    """
    Keep only top-k logits, set rest to -inf.
    Restricts sampling to the k most likely tokens.
    """
    if top_k <= 0:
        return logits
    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    min_values = values[:, -1].unsqueeze(-1)
    return logits.masked_fill(logits < min_values, float('-inf'))


def top_p_filtering(
    logits: torch.Tensor, top_p: float
) -> torch.Tensor:
    """
    Nucleus sampling: keep smallest set of tokens
    whose cumulative probability >= top_p.
    
    Dynamically adapts vocabulary size based on
    distribution shape — better than fixed top-k.
    """
    if top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(
        logits, descending=True
    )
    cumulative_probs = torch.cumsum(
        F.softmax(sorted_logits, dim=-1), dim=-1
    )

    # Remove tokens with cumulative prob above threshold
    # Shift to keep first token above threshold
    sorted_indices_to_remove = cumulative_probs - F.softmax(
        sorted_logits, dim=-1
    ) > top_p

    sorted_logits[sorted_indices_to_remove] = float('-inf')

    # Scatter back to original indexing
    logits = torch.zeros_like(logits).scatter_(
        dim=-1,
        index=sorted_indices,
        src=sorted_logits
    )
    return logits


class GreedyDecoder:
    """
    Greedy decoding: always pick the highest probability token.
    
    Pros: Fast, deterministic, no hyperparameters
    Cons: Repetitive, suboptimal for open-ended tasks
    """

    def __init__(self, config: GenerationConfig):
        self.config = config

    @torch.no_grad()
    def generate(
        self,
        model: 'GPTModel',
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from models.attention import KVCache
        config = self.config
        device = input_ids.device

        generated = input_ids.clone()
        kv_cache = KVCache()
        processors = build_logits_processors(config)

        for step in range(config.max_new_tokens):
            outputs = model(
                generated if step == 0 else generated[:, -1:],
                attention_mask=attention_mask,
                kv_cache=kv_cache,
            )

            # Get logits for last position
            logits = outputs.logits[:, -1, :]

            # Apply processors
            logits = processors(generated, logits)

            # Greedy: argmax
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)

            # Update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        device=device, dtype=attention_mask.dtype
                    )
                ], dim=-1)

            # Check EOS
            if (next_token == config.eos_token_id).all():
                break

        return generated


class SamplingDecoder:
    """
    Sampling-based decoding with top-k, top-p, temperature.
    
    Pipeline:
        logits -> temperature -> top-k -> top-p -> softmax -> sample
    """

    def __init__(self, config: GenerationConfig):
        self.config = config

    @torch.no_grad()
    def generate(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from models.attention import KVCache
        config = self.config
        device = input_ids.device

        generated = input_ids.clone()
        kv_cache = KVCache()
        processors = build_logits_processors(config)

        for step in range(config.max_new_tokens):
            outputs = model(
                generated if step == 0 else generated[:, -1:],
                attention_mask=attention_mask,
                kv_cache=kv_cache,
            )

            logits = outputs.logits[:, -1, :].float()

            # Apply processors (temp, rep penalty, etc.)
            logits = processors(generated, logits)

            # Top-k filtering
            if config.top_k > 0:
                logits = top_k_filtering(logits, config.top_k)

            # Top-p filtering
            if config.top_p < 1.0:
                logits = top_p_filtering(logits, config.top_p)

            # Sample from filtered distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=-1)

            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        device=device, dtype=attention_mask.dtype
                    )
                ], dim=-1)

            if (next_token == config.eos_token_id).all():
                break

        return generated

    @torch.no_grad()
    def generate_stream(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Iterator[torch.Tensor]:
        """Streaming generation - yield one token at a time."""
        from models.attention import KVCache
        config = self.config
        kv_cache = KVCache()
        processors = build_logits_processors(config)
        generated = input_ids.clone()

        for step in range(config.max_new_tokens):
            outputs = model(
                generated if step == 0 else generated[:, -1:],
                kv_cache=kv_cache,
            )

            logits = outputs.logits[:, -1, :].float()
            logits = processors(generated, logits)

            if config.top_k > 0:
                logits = top_k_filtering(logits, config.top_k)
            if config.top_p < 1.0:
                logits = top_p_filtering(logits, config.top_p)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)

            yield next_token

            if (next_token == config.eos_token_id).all():
                break


@dataclass
class BeamHypothesis:
    token_ids: list[int]
    score: float
    is_done: bool = False

    def __len__(self):
        return len(self.token_ids)


class BeamSearchDecoder:
    """
    Beam Search decoding.
    
    Maintains `num_beams` hypotheses in parallel.
    Explores a wider search space than greedy.
    
    Pros: More coherent, better for translation/summarization
    Cons: Slower, can be repetitive for open-ended generation
    
    Score: log P(y|x) / length_penalty
    """

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.num_beams = config.num_beams

    @torch.no_grad()
    def generate(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        config = self.config
        device = input_ids.device
        B = input_ids.shape[0]

        # Expand input for beams: (B*num_beams, S)
        input_ids_beam = input_ids.repeat_interleave(
            self.num_beams, dim=0
        )

        # Beam scores: (B, num_beams)
        beam_scores = torch.full(
            (B, self.num_beams), float('-inf'), device=device
        )
        beam_scores[:, 0] = 0.0
        beam_scores = beam_scores.view(-1)

        generated = input_ids_beam.clone()
        done_beams: list[list[BeamHypothesis]] = [
            [] for _ in range(B)
        ]
        processors = build_logits_processors(config)

        for step in range(config.max_new_tokens):
            outputs = model(
                generated if step == 0 else generated[:, -1:],
            )

            # logits: (B*num_beams, vocab_size)
            logits = outputs.logits[:, -1, :].float()
            logits = processors(generated, logits)

            log_probs = F.log_softmax(logits, dim=-1)
            vocab_size = log_probs.shape[-1]

            # Add beam scores
            next_scores = log_probs + beam_scores.unsqueeze(-1)

            # Reshape: (B, num_beams * vocab_size)
            next_scores = next_scores.view(B, self.num_beams * vocab_size)

            # Top-(2*num_beams) candidates per batch
            top_scores, top_indices = torch.topk(
                next_scores, 2 * self.num_beams, dim=-1
            )

            next_beam_tokens = top_indices % vocab_size
            next_beam_indices = top_indices // vocab_size

            new_generated = []
            new_scores = []

            for batch_idx in range(B):
                beam_hyps = []
                for rank in range(2 * self.num_beams):
                    beam_idx = next_beam_indices[batch_idx, rank]
                    token_id = next_beam_tokens[batch_idx, rank]
                    score = top_scores[batch_idx, rank].item()

                    # Get current sequence for this beam
                    flat_idx = batch_idx * self.num_beams + beam_idx
                    seq = generated[flat_idx].tolist()
                    seq.append(token_id.item())

                    if token_id.item() == config.eos_token_id:
                        length = len(seq)
                        length_pen = (
                            (5 + length) ** config.length_penalty
                            / (5 + 1) ** config.length_penalty
                        )
                        done_beams[batch_idx].append(
                            BeamHypothesis(
                                token_ids=seq,
                                score=score / length_pen,
                                is_done=True
                            )
                        )
                    else:
                        beam_hyps.append((score, seq))

                    if len(beam_hyps) == self.num_beams:
                        break

                for score, seq in beam_hyps:
                    new_scores.append(score)
                    new_generated.append(
                        torch.tensor(seq, device=device)
                    )

            # Pad and stack
            max_len = max(s.shape[0] for s in new_generated)
            generated = torch.stack([
                F.pad(
                    s, (0, max_len - s.shape[0]),
                    value=config.pad_token_id
                )
                for s in new_generated
            ])
            beam_scores = torch.tensor(new_scores, device=device)

            # Check early stopping
            all_done = all(
                len(done_beams[i]) >= self.num_beams
                for i in range(B)
            )
            if config.beam_search_early_stopping and all_done:
                break

        # Select best hypothesis per batch
        best_sequences = []
        for batch_idx in range(B):
            if done_beams[batch_idx]:
                best = max(
                    done_beams[batch_idx],
                    key=lambda h: h.score
                )
                best_sequences.append(
                    torch.tensor(best.token_ids, device=device)
                )
            else:
                # No completed beam, take best running beam
                best_score_idx = (
                    batch_idx * self.num_beams
                    + beam_scores[
                        batch_idx * self.num_beams:
                        (batch_idx + 1) * self.num_beams
                    ].argmax().item()
                )
                best_sequences.append(generated[best_score_idx])

        # Pad to same length
        max_len = max(s.shape[0] for s in best_sequences)
        result = torch.stack([
            F.pad(
                s, (0, max_len - s.shape[0]),
                value=config.pad_token_id
            )
            for s in best_sequences
        ])
        return result