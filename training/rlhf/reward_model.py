"""
Reward Model for RLHF.
Trained on human preference pairs (chosen vs rejected).
Architecture: LM backbone + scalar head.
Loss: Bradley-Terry preference loss.
"""

import os
import json
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from loguru import logger

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class RewardModelConfig:
    # Architecture
    backbone_path: str = "./checkpoints/sft/final"
    hidden_size: int = 768

    # Training
    batch_size: int = 8
    learning_rate: float = 1e-5
    num_epochs: int = 1
    max_length: int = 1024
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 100

    # Data
    data_path: str = "./data/preference"

    # Output
    output_dir: str = "./checkpoints/reward_model"

    # Logging
    wandb_project: str = "llm_playground"
    logging_steps: int = 10


class RewardModel(nn.Module):
    """
    Reward Model = Pretrained LM + Scalar Head.
    
    Given a (prompt, response) pair, outputs a scalar reward.
    Trained via Bradley-Terry model on preference pairs:
        L = -log(sigmoid(r_chosen - r_rejected))
    
    Higher reward = better response (as judged by humans).
    """

    def __init__(self, backbone: nn.Module, config: RewardModelConfig):
        super().__init__()
        self.backbone = backbone
        self.config = config

        # Scalar head: maps hidden_size -> 1
        self.reward_head = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

        # Initialize head
        nn.init.normal_(self.reward_head[0].weight, std=0.02)
        nn.init.zeros_(self.reward_head[0].bias)
        nn.init.normal_(self.reward_head[3].weight, std=0.02)
        nn.init.zeros_(self.reward_head[3].bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns scalar reward for each sequence.
        Uses last non-padding token's hidden state.
        
        Returns:
            rewards: (batch_size,) reward scores
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Get last hidden state: (B, S, H)
        hidden_states = outputs.hidden_states[-1]

        # Extract last real token (before padding)
        if attention_mask is not None:
            # Find last non-padding position
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_size = input_ids.shape[0]
            last_hidden = hidden_states[
                torch.arange(batch_size),
                seq_lengths
            ]
        else:
            last_hidden = hidden_states[:, -1, :]

        # Scalar reward
        reward = self.reward_head(last_hidden).squeeze(-1)
        return reward

    def compute_preference_loss(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
        chosen_mask: Optional[torch.Tensor] = None,
        rejected_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Bradley-Terry preference loss:
        L = -E[log(sigmoid(r_w - r_l))]
        where r_w = reward of chosen, r_l = reward of rejected.
        """
        chosen_reward = self(chosen_ids, chosen_mask)
        rejected_reward = self(rejected_ids, rejected_mask)

        # Bradley-Terry loss
        loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()

        # Accuracy: fraction where chosen > rejected
        accuracy = (chosen_reward > rejected_reward).float().mean()

        # Reward margin
        margin = (chosen_reward - rejected_reward).mean()

        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "chosen_reward_mean": chosen_reward.mean().item(),
            "rejected_reward_mean": rejected_reward.mean().item(),
            "reward_margin": margin.item(),
        }
        return loss, metrics


class PreferenceDataset(Dataset):
    """
    Human preference dataset.
    Each example: (prompt, chosen_response, rejected_response).
    Tokenizes and concatenates prompt + response.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 1024,
        split: str = "train"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self._load(data_path, split)
        logger.info(
            f"Preference Dataset [{split}]: {len(self.examples)}"
        )

    def _load(self, data_path: str, split: str) -> list[dict]:
        filepath = os.path.join(data_path, f"{split}.jsonl")
        examples = []

        if not os.path.exists(filepath):
            logger.warning(
                f"No preference data at {filepath}. Demo mode."
            )
            return self._demo_data()

        with open(filepath, encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    examples.append(item)
                except Exception:
                    continue
        return examples

    def _demo_data(self) -> list[dict]:
        return [
            {
                "prompt": "What is 2+2?",
                "chosen": "2+2 equals 4.",
                "rejected": "I don't know math."
            },
            {
                "prompt": "Explain gravity.",
                "chosen": (
                    "Gravity is a fundamental force that attracts "
                    "objects with mass toward each other."
                ),
                "rejected": "Things fall down because of magic."
            },
        ]

    def _encode_pair(
        self, prompt: str, response: str
    ) -> tuple[list[int], list[int]]:
        """Encode prompt + response into token ids."""
        text = f"Human: {prompt}\nAssistant: {response}"
        ids = self.tokenizer.encode(
            text, add_special_tokens=True
        )[:self.max_length]
        mask = [1] * len(ids)

        # Pad
        pad_len = self.max_length - len(ids)
        ids = ids + [0] * pad_len
        mask = mask + [0] * pad_len

        return ids[:self.max_length], mask[:self.max_length]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        item = self.examples[idx]
        prompt = item.get("prompt", "")
        chosen = item.get("chosen", "")
        rejected = item.get("rejected", "")

        chosen_ids, chosen_mask = self._encode_pair(prompt, chosen)
        rejected_ids, rejected_mask = self._encode_pair(
            prompt, rejected
        )

        return {
            "chosen_input_ids": torch.tensor(
                chosen_ids, dtype=torch.long
            ),
            "chosen_attention_mask": torch.tensor(
                chosen_mask, dtype=torch.long
            ),
            "rejected_input_ids": torch.tensor(
                rejected_ids, dtype=torch.long
            ),
            "rejected_attention_mask": torch.tensor(
                rejected_mask, dtype=torch.long
            ),
        }


class RewardModelTrainer:
    """Train the reward model on preference pairs."""

    def __init__(
        self,
        reward_model: RewardModel,
        config: RewardModelConfig
    ):
        self.model = reward_model
        self.config = config
        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.model = self.model.to(self.device)
        os.makedirs(config.output_dir, exist_ok=True)

        if WANDB_AVAILABLE:
            wandb.init(
                project=config.wandb_project,
                name="reward_model",
                config=vars(config)
            )

    def train(
        self,
        train_dataset: PreferenceDataset,
        val_dataset: Optional[PreferenceDataset] = None
    ):
        config = self.config
        loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        total_steps = len(loader) * config.num_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps
        )

        global_step = 0
        best_val_acc = 0.0

        autocast_ctx = torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=torch.cuda.is_available()
        )

        for epoch in range(config.num_epochs):
            self.model.train()

            for batch in loader:
                chosen_ids = batch["chosen_input_ids"].to(self.device)
                chosen_mask = batch["chosen_attention_mask"].to(
                    self.device
                )
                rejected_ids = batch["rejected_input_ids"].to(
                    self.device
                )
                rejected_mask = batch["rejected_attention_mask"].to(
                    self.device
                )

                with autocast_ctx:
                    loss, metrics = self.model.compute_preference_loss(
                        chosen_ids, rejected_ids,
                        chosen_mask, rejected_mask
                    )

                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), config.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % config.logging_steps == 0:
                    logger.info(
                        f"RM Step {global_step} | "
                        f"Loss: {metrics['loss']:.4f} | "
                        f"Acc: {metrics['accuracy']:.3f} | "
                        f"Margin: {metrics['reward_margin']:.4f}"
                    )
                    if WANDB_AVAILABLE:
                        wandb.log(
                            {f"rm/{k}": v for k, v in metrics.items()}
                        )

            if val_dataset:
                val_metrics = self._evaluate(val_dataset)
                logger.info(
                    f"Epoch {epoch+1} Val | "
                    f"Acc: {val_metrics['accuracy']:.3f}"
                )
                if val_metrics["accuracy"] > best_val_acc:
                    best_val_acc = val_metrics["accuracy"]
                    self._save("best")

        self._save("final")
        if WANDB_AVAILABLE:
            wandb.finish()

    def _evaluate(self, val_dataset: PreferenceDataset) -> dict:
        loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size
        )
        self.model.eval()
        all_metrics = []

        with torch.no_grad():
            for batch in loader:
                chosen_ids = batch["chosen_input_ids"].to(self.device)
                rejected_ids = batch["rejected_input_ids"].to(self.device)
                _, metrics = self.model.compute_preference_loss(
                    chosen_ids, rejected_ids
                )
                all_metrics.append(metrics)
                if len(all_metrics) >= 20:
                    break

        self.model.train()
        return {
            k: sum(m[k] for m in all_metrics) / len(all_metrics)
            for k in all_metrics[0]
        }

    def _save(self, tag: str):
        path = os.path.join(self.config.output_dir, tag)
        os.makedirs(path, exist_ok=True)
        torch.save(
            self.model.state_dict(),
            os.path.join(path, "reward_model.pt")
        )
        logger.info(f"Reward model saved: {path}")