"""
Byte Pair Encoding (BPE) tokenizer implementation.
Production-grade with full train/encode/decode support,
special tokens, and HuggingFace compatibility.
"""

import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger
from tqdm import tqdm


@dataclass
class BPEConfig:
    vocab_size: int = 50257
    min_frequency: int = 2
    special_tokens: dict = field(default_factory=lambda: {
        'pad_token': '<|pad|>',
        'eos_token': '<|endoftext|>',
        'bos_token': '<|startoftext|>',
        'unk_token': '<|unk|>',
        'sep_token': '<|sep|>',
        'mask_token': '<|mask|>',
    })
    pre_tokenize_pattern: str = (
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+"""
    )
    byte_level: bool = True


class ByteLevelBPETokenizer:
    """
    Full Byte-Level BPE Tokenizer (GPT-2 style).
    
    Pipeline:
    1. Pre-tokenization (regex splitting)
    2. Byte-level encoding
    3. BPE merges
    4. Vocabulary management
    
    Features:
    - Full UTF-8 support via byte-level encoding
    - HuggingFace-compatible save/load
    - Special token handling
    - Efficient encoding/decoding
    """

    def __init__(self, config: BPEConfig):
        self.config = config
        self.encoder: dict[str, int] = {}
        self.decoder: dict[int, str] = {}
        self.bpe_ranks: dict[tuple[str, str], int] = {}
        self.special_tokens: dict[str, int] = {}
        self.cache: dict[str, list[str]] = {}
        self._pre_tokenize = re.compile(
            config.pre_tokenize_pattern
        )
        self._init_byte_encoder()

    def _init_byte_encoder(self):
        """
        Create byte-to-unicode mapping.
        Maps each byte (0-255) to a printable unicode character.
        Avoids whitespace/control characters in vocabulary.
        """
        bs = (
            list(range(ord('!'), ord('~') + 1))
            + list(range(ord('¡'), ord('¬') + 1))
            + list(range(ord('®'), ord('ÿ') + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1

        self.byte_encoder = {b: chr(c) for b, c in zip(bs, cs)}
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

    def _pre_tokenize_text(self, text: str) -> list[str]:
        return re.findall(self._pre_tokenize, text)

    def _word_to_chars(self, word: str) -> tuple[str, ...]:
        """Convert word bytes to unicode chars."""
        return tuple(
            self.byte_encoder[b] for b in word.encode('utf-8')
        )

    def _get_pairs(
        self, word: tuple[str, ...]
    ) -> set[tuple[str, str]]:
        return {
            (word[i], word[i + 1])
            for i in range(len(word) - 1)
        }

    def _bpe_encode(self, token: str) -> list[str]:
        """Apply BPE merges to a single token."""
        if token in self.cache:
            return self.cache[token]

        word = self._word_to_chars(token)
        if len(word) == 1:
            return list(word)

        pairs = self._get_pairs(word)

        while True:
            if not pairs:
                break
            bigram = min(
                pairs,
                key=lambda p: self.bpe_ranks.get(p, float('inf'))
            )
            if bigram not in self.bpe_ranks:
                break

            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = self._get_pairs(word)

        result = list(word)
        self.cache[token] = result
        return result

    def _get_word_frequencies(
        self,
        texts: list[str]
    ) -> Counter:
        """Count word frequencies from corpus."""
        word_freq: Counter = Counter()
        for text in tqdm(texts, desc="Computing frequencies"):
            tokens = self._pre_tokenize_text(text)
            for token in tokens:
                chars = self._word_to_chars(token)
                word_freq[' '.join(chars)] += 1
        return word_freq

    def _compute_pair_frequencies(
        self,
        vocab: dict[str, int]
    ) -> Counter:
        """Count adjacent pair frequencies."""
        pairs: Counter = Counter()
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def _merge_vocab(
        self,
        vocab: dict[str, int],
        pair: tuple[str, str]
    ) -> dict[str, int]:
        """Merge a pair in all vocabulary words."""
        new_vocab = {}
        bigram = re.escape(' '.join(pair))
        pattern = re.compile(
            r'(?<!\S)' + bigram + r'(?!\S)'
        )
        for word, freq in vocab.items():
            new_word = pattern.sub(''.join(pair), word)
            new_vocab[new_word] = freq
        return new_vocab

    def train(self, texts: list[str]) -> 'ByteLevelBPETokenizer':
        """
        Train BPE tokenizer on corpus.
        
        Args:
            texts: List of training texts
            
        Returns:
            self (for chaining)
        """
        logger.info(
            f"Training BPE tokenizer. "
            f"Target vocab: {self.config.vocab_size}"
        )

        # Initialize with special tokens
        vocab: dict[str, int] = {}
        for token_name, token_str in (
            self.config.special_tokens.items()
        ):
            idx = len(self.encoder)
            self.encoder[token_str] = idx
            self.decoder[idx] = token_str
            self.special_tokens[token_str] = idx
            logger.info(f"Special token: {token_str} -> {idx}")

        # Add byte-level base tokens (256 bytes)
        for i in range(256):
            char = self.byte_encoder[i]
            if char not in self.encoder:
                idx = len(self.encoder)
                self.encoder[char] = idx
                self.decoder[idx] = char

        # Count word frequencies
        word_freq = self._get_word_frequencies(texts)

        # Filter by minimum frequency
        vocab = {
            word: freq
            for word, freq in word_freq.items()
            if freq >= self.config.min_frequency
        }
        logger.info(f"Initial vocabulary types: {len(vocab)}")

        # BPE merge loop
        num_merges = self.config.vocab_size - len(self.encoder)
        logger.info(f"Running {num_merges} BPE merges")

        merges: list[tuple[str, str]] = []

        with tqdm(total=num_merges, desc="BPE Merges") as pbar:
            for merge_idx in range(num_merges):
                if not vocab:
                    break

                pairs = self._compute_pair_frequencies(vocab)
                if not pairs:
                    break

                best_pair = max(pairs, key=pairs.get)
                best_freq = pairs[best_pair]

                if best_freq < self.config.min_frequency:
                    break

                merges.append(best_pair)
                self.bpe_ranks[best_pair] = merge_idx

                merged_token = ''.join(best_pair)
                idx = len(self.encoder)
                self.encoder[merged_token] = idx
                self.decoder[idx] = merged_token

                vocab = self._merge_vocab(vocab, best_pair)
                pbar.update(1)

        logger.info(
            f"Training complete. Vocab size: {len(self.encoder)}"
        )
        return self

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True
    ) -> list[int]:
        """Encode text to token IDs."""
        bos = self.config.special_tokens.get('bos_token', '')
        eos = self.config.special_tokens.get('eos_token', '')

        token_ids = []

        if add_special_tokens and bos in self.encoder:
            token_ids.append(self.encoder[bos])

        tokens = self._pre_tokenize_text(text)
        for token in tokens:
            bpe_tokens = self._bpe_encode(token)
            for bpe_token in bpe_tokens:
                if bpe_token in self.encoder:
                    token_ids.append(self.encoder[bpe_token])
                else:
                    unk = self.config.special_tokens.get('unk_token')
                    if unk and unk in self.encoder:
                        token_ids.append(self.encoder[unk])

        if add_special_tokens and eos in self.encoder:
            token_ids.append(self.encoder[eos])

        return token_ids

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True
    ) -> str:
        """Decode token IDs back to text."""
        tokens = []
        special_ids = set(self.special_tokens.values())

        for token_id in token_ids:
            if token_id not in self.decoder:
                continue
            if skip_special_tokens and token_id in special_ids:
                continue
            tokens.append(self.decoder[token_id])

        text = ''.join(tokens)
        bytes_output = bytearray(
            self.byte_decoder.get(c, ord(c)) for c in text
        )
        return bytes_output.decode('utf-8', errors='replace')

    def encode_batch(
        self,
        texts: list[str],
        padding: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = True
    ) -> dict:
        """Batch encoding with padding and truncation."""
        all_ids = [self.encode(t) for t in texts]
        pad_id = self.encoder.get(
            self.config.special_tokens.get('pad_token', '<|pad|>'), 0
        )

        if max_length and truncation:
            all_ids = [ids[:max_length] for ids in all_ids]

        if padding:
            max_len = max(len(ids) for ids in all_ids) if all_ids else 0
            attention_masks = []
            padded_ids = []

            for ids in all_ids:
                pad_len = max_len - len(ids)
                attention_masks.append(
                    [1] * len(ids) + [0] * pad_len
                )
                padded_ids.append(ids + [pad_id] * pad_len)

            return {
                'input_ids': padded_ids,
                'attention_mask': attention_masks
            }

        return {'input_ids': all_ids}

    def save(self, save_dir: str):
        """Save tokenizer to directory (HuggingFace compatible)."""
        os.makedirs(save_dir, exist_ok=True)

        with open(
            os.path.join(save_dir, 'vocab.json'), 'w',
            encoding='utf-8'
        ) as f:
            json.dump(self.encoder, f, ensure_ascii=False, indent=2)

        merges = [
            f"{a} {b}"
            for (a, b) in sorted(
                self.bpe_ranks.keys(),
                key=lambda x: self.bpe_ranks[x]
            )
        ]
        with open(
            os.path.join(save_dir, 'merges.txt'), 'w',
            encoding='utf-8'
        ) as f:
            f.write('#version: 0.2\n')
            f.write('\n'.join(merges))

        tokenizer_config = {
            'tokenizer_class': 'ByteLevelBPETokenizer',
            'vocab_size': self.config.vocab_size,
            'model_max_length': 2048,
            'special_tokens': self.config.special_tokens,
            'bos_token': self.config.special_tokens.get('bos_token'),
            'eos_token': self.config.special_tokens.get('eos_token'),
            'unk_token': self.config.special_tokens.get('unk_token'),
            'pad_token': self.config.special_tokens.get('pad_token'),
        }
        with open(
            os.path.join(save_dir, 'tokenizer_config.json'), 'w',
            encoding='utf-8'
        ) as f:
            json.dump(tokenizer_config, f, indent=2)

        logger.info(
            f"Tokenizer saved to {save_dir}. "
            f"Vocab: {len(self.encoder)}"
        )

    @classmethod
    def load(
        cls, load_dir: str, config: Optional[BPEConfig] = None
    ) -> 'ByteLevelBPETokenizer':
        """Load tokenizer from directory."""
        with open(
            os.path.join(load_dir, 'vocab.json'), 'r',
            encoding='utf-8'
        ) as f:
            encoder = json.load(f)

        with open(
            os.path.join(load_dir, 'merges.txt'), 'r',
            encoding='utf-8'
        ) as f:
            merges_data = f.read().split('\n')

        merges = []
        for line in merges_data:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) == 2:
                merges.append(tuple(parts))

        if config is None:
            config_path = os.path.join(
                load_dir, 'tokenizer_config.json'
            )
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg_data = json.load(f)
            config = BPEConfig(
                vocab_size=cfg_data.get('vocab_size', 50257),
                special_tokens=cfg_data.get('special_tokens', {})
            )

        tokenizer = cls(config)
        tokenizer.encoder = encoder
        tokenizer.decoder = {v: k for k, v in encoder.items()}
        tokenizer.bpe_ranks = {
            tuple(merge): i for i, merge in enumerate(merges)
        }
        logger.info(
            f"Loaded tokenizer from {load_dir}. "
            f"Vocab: {len(encoder)}"
        )
        return tokenizer

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    def __len__(self) -> int:
        return self.vocab_size