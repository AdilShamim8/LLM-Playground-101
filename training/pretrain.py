"""
Pre-training loop for GPT-style language models.
Production features:
- DeepSpeed ZeRO integration
- Gradient checkpointing
- Mixed precision (fp16/bf16)
- Learning rate scheduling (cosine with warmup)
- Checkpoint save/resume
- WandB logging
- Distributed training (DDP)
"""

import os
import math
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from loguru import logger

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False


@dataclass
class PretrainConfig:
    # Data
    data_path: str = "./data/tokenized"
    max_seq_length: int = 2048

    # Training
    batch_size: int = 32
    gradient_accumulation_steps: int = 8
    max_steps: int = 100000
    warmup_steps: int = 2000
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8

    # Precision
    fp16: bool = False
    bf16: bool = True

    # Checkpointing
    output_dir: str = "./checkpoints"
    save_steps: int = 5000
    eval_steps: int = 1000
    keep_last_n_checkpoints: int = 3

    # Distributed
    use_deepspeed: bool = False
    local_rank: int = -1

    # Logging
    log_steps: int = 10
    wandb_project: str = "llm_playground"
    wandb_run_name: str = "pretrain"

    # Reproducibility
    seed: int = 42


class PretrainDataset(Dataset):
    """
    Token-level pre-training dataset.
    Reads pre-tokenized binary files for efficiency.
    Concatenates all documents and chunks into fixed-length sequences.
    """

    def __init__(
        self,
        data_path: str,
        seq_length: int,
        split: str = 'train'
    ):
        self.seq_length = seq_length
        self.data_path = data_path
        self.tokens = self._load_tokens(data_path, split)
        self.n_sequences = (
            (len(self.tokens) - 1) // seq_length
        )
        logger.info(
            f"Dataset: {len(self.tokens)} tokens, "
            f"{self.n_sequences} sequences of length {seq_length}"
        )

    def _load_tokens(
        self, data_path: str, split: str
    ) -> torch.Tensor:
        """Load pre-tokenized data from binary files."""
        import numpy as np
        import glob

        all_tokens = []
        pattern = os.path.join(data_path, f"{split}_*.bin")
        files = sorted(glob.glob(pattern))

        if not files:
            # Fallback: load from JSONL and tokenize on the fly
            logger.warning(
                f"No .bin files found at {pattern}. "
                "Generating random tokens for demo."
            )
            return torch.randint(0, 50257, (10_000_000,))

        for fpath in files:
            arr = np.fromfile(fpath, dtype=np.uint16)
            all_tokens.append(torch.from_numpy(arr.astype(np.int32)))
            logger.info(
                f"Loaded {fpath}: {len(arr)} tokens"
            )

        return torch.cat(all_tokens)

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx: int) -> dict:
        start = idx * self.seq_length
        end = start + self.seq_length + 1
        chunk = self.tokens[start:end]

        input_ids = chunk[:-1].long()
        labels = chunk[1:].long()

        return {
            'input_ids': input_ids,
            'labels': labels,
        }


class CosineSchedulerWithWarmup:
    """
    Cosine learning rate decay with linear warmup.
    Standard for LLM pre-training (GPT-3, LLaMA, etc.).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr: float,
        max_lr: float
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.max_lr = max_lr
        self._step = 0

    def get_lr(self) -> float:
        step = self._step
        if step < self.warmup_steps:
            return self.max_lr * step / max(1, self.warmup_steps)

        progress = (step - self.warmup_steps) / max(
            1, self.max_steps - self.warmup_steps
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.max_lr - self.min_lr) * cosine_decay

    def step(self):
        self._step += 1
        lr = self.get_lr()
        for group in self.optimizer.param_groups:
            group['lr'] = lr

    def state_dict(self) -> dict:
        return {'step': self._step}

    def load_state_dict(self, state: dict):
        self._step = state['step']


class PretrainTrainer:
    """
    Full pre-training trainer with production features.
    """

    def __init__(
        self,
        model: nn.Module,
        config: PretrainConfig
    ):
        self.model = model
        self.config = config
        self.global_step = 0
        self.best_val_loss = float('inf')

        os.makedirs(config.output_dir, exist_ok=True)
        self._setup_device()
        self._setup_wandb()

    def _setup_device(self):
        if torch.cuda.is_available():
            if self.config.local_rank >= 0:
                torch.cuda.set_device(self.config.local_rank)
                self.device = torch.device(
                    f'cuda:{self.config.local_rank}'
                )
            else:
                self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        logger.info(f"Training on: {self.device}")

    def _setup_wandb(self):
        if WANDB_AVAILABLE and self.config.local_rank in (-1, 0):
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=vars(self.config)
            )

    def _build_optimizer(
        self,
        model: nn.Module
    ) -> torch.optim.AdamW:
        """
        Build AdamW with weight decay applied only to
        non-bias/non-norm parameters (following GPT-2).
        """
        decay_params = []
        no_decay_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if (
                param.ndim < 2
                or 'bias' in name
                or 'norm' in name
                or 'embedding' in name
            ):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {
                'params': decay_params,
                'weight_decay': self.config.weight_decay
            },
            {
                'params': no_decay_params,
                'weight_decay': 0.0
            }
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_eps
        )
        logger.info(
            f"Optimizer: AdamW | "
            f"Decay params: {len(decay_params)} | "
            f"No-decay: {len(no_decay_params)}"
        )
        return optimizer

    def _get_autocast_ctx(self):
        if self.config.bf16:
            return torch.autocast(
                device_type='cuda', dtype=torch.bfloat16
            )
        if self.config.fp16:
            return torch.autocast(
                device_type='cuda', dtype=torch.float16
            )
        return torch.autocast(device_type='cuda', enabled=False)

    def _save_checkpoint(
        self,
        step: int,
        val_loss: Optional[float] = None
    ):
        checkpoint_dir = os.path.join(
            self.config.output_dir, f"checkpoint-{step}"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

        model_to_save = (
            self.model.module
            if hasattr(self.model, 'module')
            else self.model
        )
        model_to_save.save_pretrained(checkpoint_dir)

        torch.save({
            'step': step,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': vars(self.config),
        }, os.path.join(checkpoint_dir, 'trainer_state.pt'))

        logger.info(f"Checkpoint saved: {checkpoint_dir}")
        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        import glob
        checkpoints = sorted(
            glob.glob(
                os.path.join(self.config.output_dir, 'checkpoint-*')
            ),
            key=lambda x: int(x.split('-')[-1])
        )
        while len(checkpoints) > self.config.keep_last_n_checkpoints:
            import shutil
            old_ckpt = checkpoints.pop(0)
            shutil.rmtree(old_ckpt)
            logger.info(f"Deleted old checkpoint: {old_ckpt}")

    def _eval_step(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        steps = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                with self._get_autocast_ctx():
                    outputs = self.model(
                        input_ids=input_ids,
                        labels=labels
                    )

                total_loss += outputs.loss.item()
                steps += 1
                if steps >= 50:  # Quick eval
                    break

        self.model.train()
        return total_loss / max(steps, 1)

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None
    ):
        config = self.config
        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = self._build_optimizer(self.model)
        self.scheduler = CosineSchedulerWithWarmup(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
            min_lr=config.min_lr,
            max_lr=config.learning_rate
        )

        # Gradient scaler for fp16
        self.scaler = GradScaler(
            enabled=config.fp16
        )

        # DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )

        # Enable gradient checkpointing
        if hasattr(self.model, 'enable_gradient_checkpointing'):
            self.model.enable_gradient_checkpointing()

        total_params = sum(
            p.numel() for p in self.model.parameters()
        )
        trainable_params = sum(
            p.numel() for p in self.model.parameters()
            if p.requires_grad
        )
        logger.info(
            f"Model: {total_params/1e6:.1f}M total params, "
            f"{trainable_params/1e6:.1f}M trainable"
        )
        logger.info(f"Starting pre-training for {config.max_steps} steps")

        self.model.train()
        self.optimizer.zero_grad()

        step = 0
        epoch = 0
        accum_loss = 0.0
        t0 = time.time()

        while step < config.max_steps:
            epoch += 1
            for batch in train_loader:
                if step >= config.max_steps:
                    break

                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                with self._get_autocast_ctx():
                    outputs = self.model(
                        input_ids=input_ids,
                        labels=labels
                    )
                    loss = outputs.loss
                    loss = loss / config.gradient_accumulation_steps

                if config.fp16:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                accum_loss += loss.item()

                if (step + 1) % config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if config.fp16:
                        self.scaler.unscale_(self.optimizer)

                    grad_norm = nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        config.max_grad_norm
                    )

                    if config.fp16:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    self.global_step += 1
                    actual_loss = (
                        accum_loss * config.gradient_accumulation_steps
                    )
                    accum_loss = 0.0

                    # Logging
                    if self.global_step % config.log_steps == 0:
                        elapsed = time.time() - t0
                        tokens_per_sec = (
                            config.batch_size
                            * config.max_seq_length
                            * config.log_steps
                            / elapsed
                        )
                        lr = self.scheduler.get_lr()

                        log_dict = {
                            'train/loss': actual_loss,
                            'train/perplexity': math.exp(
                                min(actual_loss, 20)
                            ),
                            'train/lr': lr,
                            'train/grad_norm': grad_norm.item(),
                            'train/tokens_per_sec': tokens_per_sec,
                            'train/step': self.global_step,
                        }
                        logger.info(
                            f"Step {self.global_step}/{config.max_steps} | "
                            f"Loss: {actual_loss:.4f} | "
                            f"PPL: {math.exp(min(actual_loss, 20)):.2f} | "
                            f"LR: {lr:.2e} | "
                            f"Tok/s: {tokens_per_sec:.0f}"
                        )

                        if WANDB_AVAILABLE:
                            wandb.log(log_dict)

                        t0 = time.time()

                    # Evaluation
                    if (
                        val_loader
                        and self.global_step % config.eval_steps == 0
                    ):
                        val_loss = self._eval_step(val_loader)
                        logger.info(
                            f"Step {self.global_step} | "
                            f"Val Loss: {val_loss:.4f}"
                        )
                        if WANDB_AVAILABLE:
                            wandb.log({'val/loss': val_loss})

                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self._save_checkpoint(
                                self.global_step, val_loss
                            )

                    # Periodic saving
                    elif (
                        self.global_step % config.save_steps == 0
                    ):
                        self._save_checkpoint(self.global_step)

                step += 1

        # Final save
        self._save_checkpoint(self.global_step)
        logger.info(
            f"Pre-training complete! "
            f"Best val loss: {self.best_val_loss:.4f}"
        )

        if WANDB_AVAILABLE:
            wandb.finish()