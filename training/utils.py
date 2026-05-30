"""
Training utility functions.
Data collators, batch samplers, checkpoint utilities,
and training helpers shared across trainers.
"""

import os
import json
import glob
import random
import math
from dataclasses import dataclass
from typing import Optional, Iterator

import torch
import torch.nn as nn
from torch.utils.data import Dataset, Sampler
from loguru import logger


# ── Data Collators ────────────────────────────────────────────────

@dataclass
class DataCollatorForCausalLM:
    """
    Collates batches for causal language modeling.
    Handles padding and label masking for variable-length sequences.
    """
    pad_token_id: int = 0
    max_length: int = 2048
    label_pad_token_id: int = -100

    def __call__(
        self, features: list[dict]
    ) -> dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        labels = [f.get("labels", f["input_ids"]) for f in features]
        attention_masks = [
            f.get("attention_mask", torch.ones_like(f["input_ids"]))
            for f in features
        ]

        # Pad to longest in batch
        max_len = min(
            max(x.shape[0] for x in input_ids),
            self.max_length
        )

        padded_input_ids = []
        padded_labels = []
        padded_masks = []

        for ids, lbls, mask in zip(
            input_ids, labels, attention_masks
        ):
            # Truncate
            ids = ids[:max_len]
            lbls = lbls[:max_len]
            mask = mask[:max_len]

            pad_len = max_len - ids.shape[0]

            padded_input_ids.append(torch.cat([
                ids,
                torch.full((pad_len,), self.pad_token_id,
                           dtype=ids.dtype)
            ]))
            padded_labels.append(torch.cat([
                lbls,
                torch.full((pad_len,), self.label_pad_token_id,
                           dtype=lbls.dtype)
            ]))
            padded_masks.append(torch.cat([
                mask,
                torch.zeros(pad_len, dtype=mask.dtype)
            ]))

        return {
            "input_ids": torch.stack(padded_input_ids),
            "labels": torch.stack(padded_labels),
            "attention_mask": torch.stack(padded_masks),
        }


@dataclass
class DataCollatorForPreference:
    """
    Collates chosen/rejected pairs for reward model training.
    Pads both chosen and rejected to same length within batch.
    """
    pad_token_id: int = 0
    max_length: int = 1024

    def __call__(
        self, features: list[dict]
    ) -> dict[str, torch.Tensor]:
        chosen_ids = [f["chosen_input_ids"] for f in features]
        rejected_ids = [f["rejected_input_ids"] for f in features]
        chosen_masks = [
            f.get("chosen_attention_mask",
                  torch.ones_like(f["chosen_input_ids"]))
            for f in features
        ]
        rejected_masks = [
            f.get("rejected_attention_mask",
                  torch.ones_like(f["rejected_input_ids"]))
            for f in features
        ]

        def pad_batch(
            sequences: list[torch.Tensor],
            masks: list[torch.Tensor]
        ) -> tuple[torch.Tensor, torch.Tensor]:
            max_len = min(
                max(s.shape[0] for s in sequences),
                self.max_length
            )
            padded_seqs, padded_masks = [], []
            for seq, mask in zip(sequences, masks):
                seq = seq[:max_len]
                mask = mask[:max_len]
                pad_len = max_len - seq.shape[0]
                padded_seqs.append(torch.cat([
                    seq,
                    torch.full(
                        (pad_len,), self.pad_token_id,
                        dtype=seq.dtype
                    )
                ]))
                padded_masks.append(torch.cat([
                    mask,
                    torch.zeros(pad_len, dtype=mask.dtype)
                ]))
            return torch.stack(padded_seqs), torch.stack(padded_masks)

        ch_ids, ch_masks = pad_batch(chosen_ids, chosen_masks)
        rej_ids, rej_masks = pad_batch(rejected_ids, rejected_masks)

        return {
            "chosen_input_ids": ch_ids,
            "chosen_attention_mask": ch_masks,
            "rejected_input_ids": rej_ids,
            "rejected_attention_mask": rej_masks,
        }


# ── Samplers ──────────────────────────────────────────────────────

class LengthGroupedSampler(Sampler):
    """
    Groups samples by length to minimize padding waste.
    Improves training efficiency by ~10-20%.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        lengths: Optional[list[int]] = None,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed

        if lengths is None:
            lengths = self._estimate_lengths()
        self.lengths = lengths
        self._epoch = 0

    def _estimate_lengths(self) -> list[int]:
        """Estimate sequence lengths from dataset."""
        lengths = []
        for i in range(len(self.dataset)):
            try:
                item = self.dataset[i]
                ids = item.get("input_ids", [])
                lengths.append(
                    len(ids) if hasattr(ids, "__len__") else 512
                )
            except Exception:
                lengths.append(512)
        return lengths

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1

        indices = list(range(len(self.dataset)))
        # Sort by length within megabatches
        megabatch_size = self.batch_size * 1000
        megabatches = [
            indices[i:i + megabatch_size]
            for i in range(0, len(indices), megabatch_size)
        ]

        result = []
        for megabatch in megabatches:
            sorted_batch = sorted(
                megabatch, key=lambda i: self.lengths[i]
            )
            # Shuffle batches within megabatch
            batches = [
                sorted_batch[j:j + self.batch_size]
                for j in range(0, len(sorted_batch), self.batch_size)
            ]
            rng.shuffle(batches)
            for batch in batches:
                result.extend(batch)

        return iter(result)


# ── Checkpoint Utilities ──────────────────────────────────────────

def save_training_state(
    output_dir: str,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    metrics: dict,
    keep_last_n: int = 3,
):
    """
    Save full training state for resumption.
    Handles DDP-wrapped models automatically.
    """
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Unwrap DDP if needed
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )

    # Save model
    if hasattr(model_to_save, "save_pretrained"):
        model_to_save.save_pretrained(checkpoint_dir)
    else:
        torch.save(
            model_to_save.state_dict(),
            os.path.join(checkpoint_dir, "model.pt")
        )

    # Save optimizer and scheduler
    torch.save({
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scheduler": (
            scheduler.state_dict()
            if hasattr(scheduler, "state_dict") else {}
        ),
        "metrics": metrics,
    }, os.path.join(checkpoint_dir, "trainer_state.pt"))

    # Save metrics
    with open(
        os.path.join(checkpoint_dir, "metrics.json"), "w"
    ) as f:
        json.dump({"step": step, **metrics}, f, indent=2)

    logger.info(f"Checkpoint saved: {checkpoint_dir}")
    _cleanup_checkpoints(output_dir, keep_last_n)
    return checkpoint_dir


def load_training_state(
    checkpoint_dir: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    map_location: str = "cpu",
) -> dict:
    """Resume training from a checkpoint directory."""
    state_path = os.path.join(checkpoint_dir, "trainer_state.pt")
    if not os.path.exists(state_path):
        raise FileNotFoundError(
            f"No trainer state at {state_path}"
        )

    # Load model weights
    model_path = os.path.join(checkpoint_dir, "model.pt")
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=map_location)
        model.load_state_dict(state_dict)
        logger.info(f"Model loaded from {model_path}")
    elif hasattr(model, "from_pretrained"):
        model = model.__class__.from_pretrained(
            checkpoint_dir, map_location=map_location
        )
        logger.info(f"Model loaded via from_pretrained")

    # Load optimizer/scheduler state
    state = torch.load(state_path, map_location=map_location)

    if optimizer and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
        logger.info("Optimizer state restored")

    if scheduler and "scheduler" in state:
        if hasattr(scheduler, "load_state_dict"):
            scheduler.load_state_dict(state["scheduler"])
        logger.info("Scheduler state restored")

    logger.info(
        f"Resumed from step {state.get('step', 0)}"
    )
    return state


def _cleanup_checkpoints(output_dir: str, keep_last_n: int):
    """Remove old checkpoints, keeping last N."""
    checkpoints = sorted(
        glob.glob(os.path.join(output_dir, "checkpoint-*")),
        key=lambda x: int(x.split("-")[-1])
    )
    while len(checkpoints) > keep_last_n:
        import shutil
        old = checkpoints.pop(0)
        shutil.rmtree(old, ignore_errors=True)
        logger.debug(f"Removed old checkpoint: {old}")


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """Find the latest checkpoint in a directory."""
    checkpoints = glob.glob(
        os.path.join(output_dir, "checkpoint-*")
    )
    if not checkpoints:
        return None
    return max(
        checkpoints,
        key=lambda x: int(x.split("-")[-1])
    )


# ── Training Helpers ──────────────────────────────────────────────

def count_parameters(model: nn.Module) -> dict:
    """Count trainable and total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "total_M": round(total / 1e6, 2),
        "trainable_M": round(trainable / 1e6, 2),
        "trainable_pct": round(100 * trainable / max(total, 1), 2),
    }


def get_grad_norm(model: nn.Module) -> float:
    """Compute gradient L2 norm."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


def freeze_model(model: nn.Module) -> nn.Module:
    """Freeze all model parameters."""
    for param in model.parameters():
        param.requires_grad = False
    return model


def unfreeze_model(model: nn.Module) -> nn.Module:
    """Unfreeze all model parameters."""
    for param in model.parameters():
        param.requires_grad = True
    return model


def freeze_layers(
    model: nn.Module,
    num_layers_to_freeze: int,
    layer_attr: str = "transformer.layers",
) -> nn.Module:
    """
    Freeze first N transformer layers.
    Useful for fine-tuning only top layers.
    """
    try:
        layers = model
        for attr in layer_attr.split("."):
            layers = getattr(layers, attr)

        for i, layer in enumerate(layers):
            if i < num_layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False
                logger.debug(f"Frozen layer {i}")

    except AttributeError as e:
        logger.warning(f"Could not freeze layers: {e}")

    return model


def compute_warmup_steps(
    num_training_steps: int,
    warmup_ratio: float = 0.03,
) -> int:
    """Compute warmup steps from ratio."""
    return max(1, int(num_training_steps * warmup_ratio))


def tokens_per_second(
    batch_size: int,
    seq_length: int,
    step_time_seconds: float,
    gradient_accumulation: int = 1,
) -> float:
    """Compute tokens per second throughput."""
    tokens = batch_size * seq_length * gradient_accumulation
    return tokens / max(step_time_seconds, 1e-9)


def estimate_training_time(
    total_steps: int,
    steps_per_second: float,
) -> str:
    """Human-readable training time estimate."""
    total_seconds = total_steps / max(steps_per_second, 1e-9)
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    return f"{hours}h {minutes}m"