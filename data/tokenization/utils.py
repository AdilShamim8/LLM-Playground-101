"""
Tokenization utilities.
Converts cleaned JSONL text data into binary token files
for efficient pre-training data loading.
Pipeline: JSONL text → tokenize → .bin files
"""

import glob
import json
import math
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from loguru import logger
from tqdm import tqdm


@dataclass
class DatasetSplits:
    train_files: list[str]
    val_files: list[str]
    test_files: list[str]
    total_tokens: int
    train_tokens: int
    val_tokens: int
    test_tokens: int


def tokenize_corpus_to_bin(
    input_path: str,
    output_dir: str,
    tokenizer,
    max_tokens_per_file: int = 100_000_000,
    text_field: str = "text",
    add_eos: bool = True,
    num_workers: int = 1,
) -> dict:
    """
    Tokenize a JSONL corpus and write to binary .bin files.

    Each .bin file contains uint16 token IDs packed sequentially.
    Documents are concatenated with EOS token separator.
    Files are split when max_tokens_per_file is reached.

    Args:
        input_path: Path to JSONL file or directory of JSONL files
        output_dir: Where to write .bin files
        tokenizer: ByteLevelBPETokenizer instance
        max_tokens_per_file: Tokens per output shard
        text_field: Key in JSON dict containing text
        add_eos: Whether to append EOS token after each document
        num_workers: Parallel workers (future use)

    Returns:
        Statistics dict
    """
    os.makedirs(output_dir, exist_ok=True)

    # Gather input files
    if os.path.isdir(input_path):
        input_files = sorted(glob.glob(
            os.path.join(input_path, "*.jsonl")
        ))
    else:
        input_files = [input_path]

    if not input_files:
        raise FileNotFoundError(
            f"No JSONL files found at {input_path}"
        )

    logger.info(
        f"Tokenizing {len(input_files)} files -> {output_dir}"
    )

    eos_id = tokenizer.encoder.get(
        tokenizer.config.special_tokens.get("eos_token", ""), -1
    )

    stats = {
        "total_documents": 0,
        "total_tokens": 0,
        "num_shards": 0,
        "skipped_documents": 0,
    }

    current_shard: list[int] = []
    shard_idx = 0

    def flush_shard():
        nonlocal shard_idx, current_shard
        if not current_shard:
            return
        arr = np.array(current_shard, dtype=np.uint16)
        shard_path = os.path.join(
            output_dir, f"train_{shard_idx:05d}.bin"
        )
        arr.tofile(shard_path)
        logger.info(
            f"Shard {shard_idx}: {len(arr):,} tokens -> {shard_path}"
        )
        stats["num_shards"] += 1
        shard_idx += 1
        current_shard = []

    for input_file in input_files:
        logger.info(f"Processing: {input_file}")

        with open(input_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(tqdm(f, desc="Tokenizing")):
                line = line.strip()
                if not line:
                    continue

                try:
                    doc = json.loads(line)
                    text = doc.get(text_field, "")
                    if not text or len(text) < 10:
                        stats["skipped_documents"] += 1
                        continue

                    # Tokenize
                    token_ids = tokenizer.encode(
                        text, add_special_tokens=False
                    )

                    if not token_ids:
                        stats["skipped_documents"] += 1
                        continue

                    # Add EOS between documents
                    if add_eos and eos_id >= 0:
                        token_ids.append(eos_id)

                    # Clip to uint16 range
                    token_ids = [
                        min(t, 65535) for t in token_ids
                    ]

                    current_shard.extend(token_ids)
                    stats["total_documents"] += 1
                    stats["total_tokens"] += len(token_ids)

                    # Flush when shard is full
                    if len(current_shard) >= max_tokens_per_file:
                        flush_shard()

                except json.JSONDecodeError:
                    stats["skipped_documents"] += 1
                except Exception as e:
                    logger.warning(
                        f"Error line {line_num}: {e}"
                    )
                    stats["skipped_documents"] += 1

    # Flush remaining
    flush_shard()

    logger.info(
        f"Tokenization complete. "
        f"Docs: {stats['total_documents']:,} | "
        f"Tokens: {stats['total_tokens']:,} | "
        f"Shards: {stats['num_shards']}"
    )

    # Save metadata
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def load_tokenized_dataset(
    data_dir: str,
    split: str = "train",
) -> np.ndarray:
    """
    Load all binary token files for a split into a single array.

    Args:
        data_dir: Directory containing .bin files
        split: "train" | "val" | "test"

    Returns:
        numpy array of uint16 token IDs
    """
    pattern = os.path.join(data_dir, f"{split}_*.bin")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No {split} .bin files found in {data_dir}"
        )

    arrays = []
    total = 0
    for fpath in files:
        arr = np.fromfile(fpath, dtype=np.uint16)
        arrays.append(arr)
        total += len(arr)
        logger.debug(f"Loaded {fpath}: {len(arr):,} tokens")

    logger.info(
        f"Loaded {len(files)} {split} shards: {total:,} tokens"
    )
    return np.concatenate(arrays) if arrays else np.array([], dtype=np.uint16)


def estimate_token_count(
    input_path: str,
    tokenizer,
    sample_size: int = 1000,
    text_field: str = "text",
) -> dict:
    """
    Estimate total token count by sampling documents.
    Useful for planning training before full tokenization.

    Returns:
        dict with estimated stats
    """
    if os.path.isdir(input_path):
        files = sorted(glob.glob(os.path.join(input_path, "*.jsonl")))
    else:
        files = [input_path]

    sampled_texts = []
    total_docs_estimate = 0

    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                total_docs_estimate += 1
                if len(sampled_texts) < sample_size:
                    try:
                        doc = json.loads(line.strip())
                        text = doc.get(text_field, "")
                        if text:
                            sampled_texts.append(text)
                    except Exception:
                        pass

    if not sampled_texts:
        return {"estimated_total_tokens": 0}

    # Compute avg tokens per doc from sample
    sample_token_counts = []
    for text in sampled_texts[:100]:  # Limit actual tokenization
        ids = tokenizer.encode(text, add_special_tokens=False)
        sample_token_counts.append(len(ids))

    avg_tokens = sum(sample_token_counts) / len(sample_token_counts)
    estimated_total = int(avg_tokens * total_docs_estimate)

    return {
        "sample_size": len(sample_token_counts),
        "avg_tokens_per_doc": round(avg_tokens, 1),
        "estimated_total_docs": total_docs_estimate,
        "estimated_total_tokens": estimated_total,
        "estimated_total_tokens_billions": round(
            estimated_total / 1e9, 3
        ),
    }


def split_dataset(
    data_dir: str,
    output_dir: str,
    val_ratio: float = 0.005,
    test_ratio: float = 0.001,
    seed: int = 42,
) -> DatasetSplits:
    """
    Split tokenized .bin shards into train/val/test.

    Args:
        data_dir: Input directory with train_*.bin files
        output_dir: Where to write split files
        val_ratio: Fraction for validation
        test_ratio: Fraction for test
        seed: Random seed for reproducibility

    Returns:
        DatasetSplits with file paths and token counts
    """
    import random
    random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    all_files = sorted(glob.glob(
        os.path.join(data_dir, "train_*.bin")
    ))
    if not all_files:
        raise FileNotFoundError(
            f"No train_*.bin files found in {data_dir}"
        )

    random.shuffle(all_files)

    n_total = len(all_files)
    n_test = max(1, int(n_total * test_ratio))
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val - n_test

    train_files = all_files[:n_train]
    val_files = all_files[n_train:n_train + n_val]
    test_files = all_files[n_train + n_val:]

    def count_tokens(files: list[str]) -> int:
        total = 0
        for f in files:
            arr = np.fromfile(f, dtype=np.uint16)
            total += len(arr)
        return total

    def link_or_copy(src_files, split_name):
        """Create symlinks or copy files for split."""
        result = []
        for i, src in enumerate(src_files):
            fname = os.path.basename(src)
            dst = os.path.join(output_dir, f"{split_name}_{i:05d}.bin")
            if not os.path.exists(dst):
                try:
                    os.symlink(os.path.abspath(src), dst)
                except OSError:
                    import shutil
                    shutil.copy2(src, dst)
            result.append(dst)
        return result

    train_out = link_or_copy(train_files, "train")
    val_out = link_or_copy(val_files, "val")
    test_out = link_or_copy(test_files, "test")

    train_tokens = count_tokens(train_files)
    val_tokens = count_tokens(val_files)
    test_tokens = count_tokens(test_files)

    splits = DatasetSplits(
        train_files=train_out,
        val_files=val_out,
        test_files=test_out,
        total_tokens=train_tokens + val_tokens + test_tokens,
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        test_tokens=test_tokens,
    )

    # Save split metadata
    meta = {
        "train_shards": len(train_out),
        "val_shards": len(val_out),
        "test_shards": len(test_out),
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "test_tokens": test_tokens,
        "total_tokens": splits.total_tokens,
    }
    with open(os.path.join(output_dir, "splits.json"), "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        f"Dataset split: "
        f"Train={len(train_out)} shards ({train_tokens:,} tokens) | "
        f"Val={len(val_out)} shards ({val_tokens:,} tokens) | "
        f"Test={len(test_out)} shards ({test_tokens:,} tokens)"
    )
    return splits