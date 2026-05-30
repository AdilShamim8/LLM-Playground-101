"""
Unified generation interface.
Combines all generation strategies with a clean API.
"""

from typing import Iterator, Optional, Union

import torch

from generation.strategies import (
    BeamSearchDecoder,
    GenerationConfig,
    GreedyDecoder,
    SamplingDecoder,
)


class TextGenerator:
    """
    Unified text generator supporting all decoding strategies.
    
    Usage:
        generator = TextGenerator(model, tokenizer, config)
        text = generator.generate("Hello, my name is")
    """

    def __init__(self, model, tokenizer, config: GenerationConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(model.parameters()).device

        self.greedy = GreedyDecoder(config)
        self.sampler = SamplingDecoder(config)
        self.beam_search = BeamSearchDecoder(config)

    def _encode(self, prompt: str) -> torch.Tensor:
        token_ids = self.tokenizer.encode(
            prompt, add_special_tokens=True
        )
        return torch.tensor(
            [token_ids], dtype=torch.long, device=self.device
        )

    def _decode(self, token_ids: torch.Tensor) -> str:
        return self.tokenizer.decode(
            token_ids[0].tolist(),
            skip_special_tokens=True
        )

    def generate(
        self,
        prompt: Union[str, torch.Tensor],
        strategy: str = 'sampling',
        **kwargs
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text or tensor
            strategy: 'greedy' | 'sampling' | 'beam_search'
            **kwargs: Override generation config params
        
        Returns:
            Generated text string
        """
        self.model.eval()

        if isinstance(prompt, str):
            input_ids = self._encode(prompt)
        else:
            input_ids = prompt.to(self.device)

        # Apply config overrides
        if kwargs:
            config_dict = vars(self.config).copy()
            config_dict.update(kwargs)
            config = GenerationConfig(**config_dict)
            self.greedy = GreedyDecoder(config)
            self.sampler = SamplingDecoder(config)
            self.beam_search = BeamSearchDecoder(config)
        
        if strategy == 'greedy':
            output_ids = self.greedy.generate(
                self.model, input_ids
            )
        elif strategy == 'beam_search':
            output_ids = self.beam_search.generate(
                self.model, input_ids
            )
        else:  # sampling (default)
            output_ids = self.sampler.generate(
                self.model, input_ids
            )

        # Decode only new tokens
        new_ids = output_ids[:, input_ids.shape[1]:]
        return self._decode(new_ids)

    def generate_stream(
        self,
        prompt: str,
        strategy: str = 'sampling',
        **kwargs
    ) -> Iterator[str]:
        """
        Streaming generation — yields one decoded token at a time.
        Use for real-time chatbot responses.
        """
        self.model.eval()
        input_ids = self._encode(prompt)

        if kwargs:
            config_dict = vars(self.config).copy()
            config_dict.update({k: v for k, v in kwargs.items() if v is not None})
            config = GenerationConfig(**config_dict)
            sampler = SamplingDecoder(config)
            greedy = GreedyDecoder(config)
            beam_search = BeamSearchDecoder(config)
        else:
            sampler = self.sampler
            greedy = self.greedy
            beam_search = self.beam_search

        if strategy == 'sampling':
            for token_tensor in sampler.generate_stream(
                self.model, input_ids
            ):
                token_str = self.tokenizer.decode(
                    token_tensor[0].tolist(),
                    skip_special_tokens=True
                )
                if token_str:
                    yield token_str
            return

        if strategy == 'greedy':
            output_ids = greedy.generate(self.model, input_ids)
        else:
            output_ids = beam_search.generate(self.model, input_ids)

        new_ids = output_ids[:, input_ids.shape[1]:]
        output_text = self._decode(new_ids)
        for chunk in output_text.split():
            yield chunk + " "

    def batch_generate(
        self,
        prompts: list[str],
        strategy: str = 'sampling'
    ) -> list[str]:
        """Generate from a batch of prompts."""
        encoded = self.tokenizer.encode_batch(
            prompts, padding=True, truncation=True,
            max_length=512
        )
        input_ids = torch.tensor(
            encoded['input_ids'],
            dtype=torch.long,
            device=self.device
        )
        attention_mask = torch.tensor(
            encoded['attention_mask'],
            dtype=torch.long,
            device=self.device
        )

        self.model.eval()
        if strategy == 'greedy':
            output_ids = self.greedy.generate(
                self.model, input_ids, attention_mask
            )
        else:
            output_ids = self.sampler.generate(
                self.model, input_ids, attention_mask
            )

        results = []
        for i in range(output_ids.shape[0]):
            new_ids = output_ids[i, input_ids.shape[1]:]
            results.append(
                self.tokenizer.decode(
                    new_ids.tolist(), skip_special_tokens=True
                )
            )
        return results