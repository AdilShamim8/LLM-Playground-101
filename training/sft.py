"""
Supervised Fine-Tuning (SFT) trainer.
Production features:
- Instruction/chat template formatting
- Loss masking (only train on assistant tokens)
- LoRA / PEFT integration
- Packing for efficiency
- Multi-turn conversation support
"""

import os
import json
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from loguru import logger

try:
    from peft import (
        LoraConfig,
        get_peft_model,
        TaskType,
        prepare_model_for_kbit_training
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class SFTConfig:
    # Data
    data_path: str = "./data/sft"
    max_length: int = 2048
    pack_sequences: bool = True

    # Training
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0

    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "out_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Precision
    bf16: bool = True
    fp16: bool = False

    # Checkpointing
    output_dir: str = "./checkpoints/sft"
    save_steps: int = 500
    eval_steps: int = 200
    logging_steps: int = 10

    # Wandb
    wandb_project: str = "llm_playground"
    wandb_run_name: str = "sft"

    # Chat template
    system_prompt: str = (
        "You are a helpful, harmless, and honest AI assistant."
    )


# ─── Chat Templates ───────────────────────────────────────────────

CHAT_TEMPLATES = {
    "chatml": {
        "system_start": "<|im_start|>system\n",
        "system_end": "<|im_end|>\n",
        "user_start": "<|im_start|>user\n",
        "user_end": "<|im_end|>\n",
        "assistant_start": "<|im_start|>assistant\n",
        "assistant_end": "<|im_end|>\n",
    },
    "llama2": {
        "system_start": "[INST] <<SYS>>\n",
        "system_end": "\n<</SYS>>\n\n",
        "user_start": "",
        "user_end": " [/INST] ",
        "assistant_start": "",
        "assistant_end": " </s><s>[INST] ",
    },
    "alpaca": {
        "instruction_start": "### Instruction:\n",
        "instruction_end": "\n\n",
        "input_start": "### Input:\n",
        "input_end": "\n\n",
        "response_start": "### Response:\n",
        "response_end": "\n\n",
    },
}


class ChatFormatter:
    """
    Format conversations into training strings with
    proper templates and loss masking positions.
    """

    def __init__(
        self,
        template: str = "chatml",
        system_prompt: str = ""
    ):
        self.template = CHAT_TEMPLATES.get(template, CHAT_TEMPLATES["chatml"])
        self.system_prompt = system_prompt
        self.template_name = template

    def format_conversation(
        self,
        messages: list[dict],
        tokenizer,
        max_length: int = 2048
    ) -> dict:
        """
        Format multi-turn conversation and compute loss mask.
        Loss mask: 1 for assistant tokens, 0 for everything else.
        
        Args:
            messages: [{"role": "user/assistant/system", "content": "..."}]
            tokenizer: BPE tokenizer
            max_length: Maximum sequence length
            
        Returns:
            dict with input_ids, labels, attention_mask
        """
        tmpl = self.template
        full_text = ""
        assistant_ranges = []  # (start_char, end_char)

        # Add system message
        if self.system_prompt:
            full_text += (
                tmpl["system_start"]
                + self.system_prompt
                + tmpl["system_end"]
            )

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                full_text += (
                    tmpl["system_start"]
                    + content
                    + tmpl["system_end"]
                )
            elif role == "user":
                full_text += (
                    tmpl["user_start"]
                    + content
                    + tmpl["user_end"]
                )
            elif role == "assistant":
                start = len(full_text) + len(tmpl["assistant_start"])
                full_text += tmpl["assistant_start"]
                content_start = len(full_text)
                full_text += content + tmpl["assistant_end"]
                content_end = len(full_text)
                assistant_ranges.append((content_start, content_end))

        # Tokenize full text
        token_ids = tokenizer.encode(
            full_text, add_special_tokens=False
        )[:max_length]

        # Build labels with masking
        # We do character-level -> token-level mapping
        labels = [-100] * len(token_ids)

        # Simple heuristic: find assistant token ranges
        # by encoding prefix and suffix
        char_pos = 0
        token_pos = 0
        cumulative_ids = []

        for i, token_id in enumerate(token_ids):
            token_str = tokenizer.decode(
                [token_id], skip_special_tokens=False
            )
            cumulative_ids.append(token_id)
            decoded_so_far = tokenizer.decode(
                cumulative_ids, skip_special_tokens=False
            )
            curr_char_pos = len(decoded_so_far)

            for (a_start, a_end) in assistant_ranges:
                if char_pos >= a_start and char_pos < a_end:
                    labels[i] = token_id
                    break

            char_pos = curr_char_pos

        attention_mask = [1] * len(token_ids)

        # Pad to max_length
        pad_len = max_length - len(token_ids)
        if pad_len > 0:
            token_ids = token_ids + [0] * pad_len
            labels = labels + [-100] * pad_len
            attention_mask = attention_mask + [0] * pad_len

        return {
            "input_ids": token_ids[:max_length],
            "labels": labels[:max_length],
            "attention_mask": attention_mask[:max_length],
        }

    def format_instruction(
        self,
        instruction: str,
        response: str,
        input_text: str = "",
        tokenizer=None,
        max_length: int = 2048
    ) -> dict:
        """Format alpaca-style instruction tuning examples."""
        if self.template_name == "alpaca":
            tmpl = self.template
            text = tmpl["instruction_start"] + instruction
            text += tmpl["instruction_end"]
            if input_text:
                text += tmpl["input_start"] + input_text
                text += tmpl["input_end"]
            text += tmpl["response_start"] + response
            text += tmpl["response_end"]
        else:
            messages = [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response},
            ]
            return self.format_conversation(
                messages, tokenizer, max_length
            )

        if tokenizer:
            return self.format_conversation(
                [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": response},
                ],
                tokenizer,
                max_length
            )
        return {"text": text}


class SFTDataset(Dataset):
    """
    SFT Dataset supporting:
    - Instruction-response pairs
    - Multi-turn conversations
    - Sequence packing for efficiency
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        config: SFTConfig,
        split: str = "train"
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.formatter = ChatFormatter(
            system_prompt=config.system_prompt
        )
        self.examples = self._load_data(data_path, split)
        logger.info(
            f"SFT Dataset [{split}]: {len(self.examples)} examples"
        )

    def _load_data(
        self, data_path: str, split: str
    ) -> list[dict]:
        examples = []
        filepath = os.path.join(data_path, f"{split}.jsonl")

        if not os.path.exists(filepath):
            logger.warning(
                f"No data at {filepath}. Using demo data."
            )
            return self._demo_data()

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    processed = self._process_item(item)
                    if processed:
                        examples.append(processed)
                except Exception as e:
                    logger.warning(f"Skipping malformed line: {e}")

        return examples

    def _process_item(self, item: dict) -> Optional[dict]:
        """Process a single data item into token format."""
        # Support multiple formats
        if "messages" in item:
            # Conversational format
            return self.formatter.format_conversation(
                item["messages"],
                self.tokenizer,
                self.config.max_length
            )
        elif "instruction" in item and "output" in item:
            # Alpaca format
            return self.formatter.format_instruction(
                instruction=item["instruction"],
                response=item["output"],
                input_text=item.get("input", ""),
                tokenizer=self.tokenizer,
                max_length=self.config.max_length
            )
        elif "prompt" in item and "response" in item:
            # Simple prompt-response
            messages = [
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": item["response"]},
            ]
            return self.formatter.format_conversation(
                messages, self.tokenizer, self.config.max_length
            )
        return None

    def _demo_data(self) -> list[dict]:
        """Generate demo SFT data for testing."""
        demo_examples = [
            {
                "instruction": "What is machine learning?",
                "output": (
                    "Machine learning is a subset of artificial "
                    "intelligence that enables systems to learn "
                    "and improve from experience without being "
                    "explicitly programmed."
                )
            },
            {
                "instruction": "Write a Python function to reverse a string.",
                "output": (
                    "def reverse_string(s: str) -> str:\n"
                    "    return s[::-1]\n\n"
                    "# Example:\n"
                    "print(reverse_string('hello'))  # 'olleh'"
                )
            },
        ]
        return [
            self._process_item(ex) for ex in demo_examples
            if self._process_item(ex)
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        return {
            "input_ids": torch.tensor(
                ex["input_ids"], dtype=torch.long
            ),
            "labels": torch.tensor(
                ex["labels"], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                ex["attention_mask"], dtype=torch.long
            ),
        }


class SFTTrainer:
    """
    Supervised Fine-Tuning trainer.
    Supports full fine-tuning and LoRA.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: SFTConfig
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.global_step = 0

        os.makedirs(config.output_dir, exist_ok=True)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Apply LoRA if enabled
        if config.use_lora and PEFT_AVAILABLE:
            self.model = self._apply_lora(model)
        else:
            self.model = model

        self.model = self.model.to(self.device)

        if WANDB_AVAILABLE:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=vars(config)
            )

    def _apply_lora(self, model: nn.Module) -> nn.Module:
        """Apply LoRA adapters for parameter-efficient fine-tuning."""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        total = sum(p.numel() for p in model.parameters())
        logger.info(
            f"LoRA applied | Trainable: {trainable/1e6:.2f}M "
            f"/ {total/1e6:.2f}M "
            f"({100*trainable/total:.2f}%)"
        )
        return model

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """AdamW with cosine schedule."""
        decay_params, no_decay_params = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        return torch.optim.AdamW(
            [
                {"params": decay_params,
                 "weight_decay": self.config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )

    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Cross-entropy loss with label masking.
        Only computes loss on assistant tokens (label != -100).
        """
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="mean"
        )
        return loss

    def train(
        self,
        train_dataset: SFTDataset,
        val_dataset: Optional[SFTDataset] = None
    ):
        config = self.config

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )

        optimizer = self._build_optimizer()
        total_steps = (
            len(train_loader)
            * config.num_epochs
            // config.gradient_accumulation_steps
        )
        warmup_steps = int(total_steps * config.warmup_ratio)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
            anneal_strategy="cos",
        )

        autocast_ctx = torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16 if config.bf16 else torch.float16,
            enabled=config.bf16 or config.fp16
        )

        logger.info(
            f"SFT Training | Steps: {total_steps} | "
            f"Warmup: {warmup_steps} | "
            f"LR: {config.learning_rate}"
        )

        best_val_loss = float("inf")
        self.model.train()
        optimizer.zero_grad()

        for epoch in range(config.num_epochs):
            epoch_loss = 0.0
            t0 = time.time()

            for step, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                with autocast_ctx:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    loss = self._compute_loss(outputs.logits, labels)
                    loss = loss / config.gradient_accumulation_steps

                loss.backward()
                epoch_loss += loss.item()

                if (step + 1) % config.gradient_accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), config.max_grad_norm
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    self.global_step += 1

                    if self.global_step % config.logging_steps == 0:
                        lr = scheduler.get_last_lr()[0]
                        actual_loss = (
                            epoch_loss * config.gradient_accumulation_steps
                            / (step + 1)
                        )
                        elapsed = time.time() - t0
                        logger.info(
                            f"Epoch {epoch+1} | "
                            f"Step {self.global_step} | "
                            f"Loss: {actual_loss:.4f} | "
                            f"LR: {lr:.2e} | "
                            f"Time: {elapsed:.1f}s"
                        )
                        if WANDB_AVAILABLE:
                            wandb.log({
                                "sft/loss": actual_loss,
                                "sft/lr": lr,
                                "sft/step": self.global_step,
                            })

                    if (
                        val_dataset
                        and self.global_step % config.eval_steps == 0
                    ):
                        val_loss = self._evaluate(val_dataset)
                        logger.info(
                            f"Val Loss: {val_loss:.4f}"
                        )
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            self._save(f"best")

                    if self.global_step % config.save_steps == 0:
                        self._save(f"step-{self.global_step}")

        self._save("final")
        if WANDB_AVAILABLE:
            wandb.finish()
        logger.info(
            f"SFT complete. Best val loss: {best_val_loss:.4f}"
        )

    def _evaluate(self, val_dataset: SFTDataset) -> float:
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size
        )
        self.model.eval()
        total_loss = 0.0
        n = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                outputs = self.model(input_ids=input_ids)
                loss = self._compute_loss(outputs.logits, labels)
                total_loss += loss.item()
                n += 1
                if n >= 20:
                    break

        self.model.train()
        return total_loss / max(n, 1)

    def _save(self, tag: str):
        save_dir = os.path.join(self.config.output_dir, tag)
        os.makedirs(save_dir, exist_ok=True)
        model_to_save = (
            self.model.module
            if hasattr(self.model, "module")
            else self.model
        )
        if PEFT_AVAILABLE and hasattr(model_to_save, "save_pretrained"):
            model_to_save.save_pretrained(save_dir)
        else:
            torch.save(
                model_to_save.state_dict(),
                os.path.join(save_dir, "model.pt")
            )
        logger.info(f"SFT model saved: {save_dir}")