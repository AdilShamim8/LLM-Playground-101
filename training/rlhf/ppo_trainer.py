"""
Proximal Policy Optimization (PPO) for RLHF.
Full implementation with:
- Actor (policy) and Critic (value) networks
- KL penalty from reference model
- GAE advantage estimation
- PPO clipping objective
- Mini-batch updates
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class PPOConfig:
    # PPO Hyperparameters
    ppo_epochs: int = 4
    batch_size: int = 64
    mini_batch_size: int = 8
    gamma: float = 1.0           # Discount factor
    lam: float = 0.95            # GAE lambda
    clip_range: float = 0.2      # PPO clip epsilon
    vf_coef: float = 0.1         # Value function loss weight
    ent_coef: float = 0.0        # Entropy bonus weight
    max_grad_norm: float = 1.0

    # KL Divergence from reference model
    kl_coef: float = 0.1         # Initial KL coefficient
    target_kl: float = 6.0       # Target KL for adaptive control
    kl_horizon: int = 10000      # Steps for KL controller

    # Learning rates
    actor_lr: float = 1.4e-5
    critic_lr: float = 1.4e-5

    # Generation
    max_new_tokens: int = 256
    temperature: float = 1.0

    # Reward normalization
    normalize_rewards: bool = True
    reward_clip: float = 10.0

    # Logging
    output_dir: str = "./checkpoints/ppo"
    wandb_project: str = "llm_playground"
    log_steps: int = 10
    save_steps: int = 100
    total_steps: int = 1000


class ValueHead(nn.Module):
    """
    Value head for the critic (V-function).
    Predicts expected cumulative reward from current state.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.zeros_(layer.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, S, H) last hidden states
        Returns:
            values: (B, S) value estimates
        """
        return self.head(hidden_states).squeeze(-1)


class AdaptiveKLController:
    """
    Adaptive KL divergence controller.
    Adjusts KL coefficient to maintain target KL.
    From Ziegler et al. (2019) InstructGPT paper.
    """

    def __init__(
        self,
        init_kl_coef: float,
        target_kl: float,
        horizon: int
    ):
        self.kl_coef = init_kl_coef
        self.target_kl = target_kl
        self.horizon = horizon

    def update(self, current_kl: float, n_steps: int):
        """
        Adjust KL coefficient based on measured KL vs target.
        Uses proportional controller.
        """
        proportional_error = (
            (current_kl / self.target_kl) - 1
        )
        mult = 1.0 + proportional_error * n_steps / self.horizon
        self.kl_coef = max(0.001, self.kl_coef * mult)
        return self.kl_coef


@dataclass
class PPORollout:
    """Stores a single PPO rollout (trajectory)."""
    query_ids: torch.Tensor
    response_ids: torch.Tensor
    log_probs: torch.Tensor     # (S,) log probs of generated tokens
    values: torch.Tensor        # (S,) value estimates
    rewards: torch.Tensor       # (S,) per-token rewards (last = RM score)
    advantages: torch.Tensor = None
    returns: torch.Tensor = None


class PPOTrainer:
    """
    Full PPO trainer for RLHF.
    
    Training loop:
    1. Sample prompts from dataset
    2. Generate responses with policy (actor)
    3. Score responses with reward model
    4. Compute advantages via GAE
    5. Update actor with PPO clipping
    6. Update critic with MSE loss
    7. Apply KL penalty from reference model
    
    Reference: "Training language models to follow instructions
                with human feedback" (InstructGPT, 2022)
    """

    def __init__(
        self,
        actor: nn.Module,
        critic_backbone: nn.Module,
        ref_model: nn.Module,
        reward_model: nn.Module,
        tokenizer,
        config: PPOConfig
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        # Actor (policy model)
        self.actor = actor.to(self.device)

        # Critic = backbone + value head
        self.value_head = ValueHead(
            critic_backbone.config.hidden_size
        ).to(self.device)
        self.critic_backbone = critic_backbone.to(self.device)

        # Reference model (frozen SFT model)
        self.ref_model = ref_model.to(self.device)
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()

        # Reward model (frozen)
        self.reward_model = reward_model.to(self.device)
        for param in self.reward_model.parameters():
            param.requires_grad = False
        self.reward_model.eval()

        # Optimizers
        self.actor_optimizer = torch.optim.AdamW(
            [p for p in self.actor.parameters() if p.requires_grad],
            lr=config.actor_lr,
            eps=1e-5
        )
        self.critic_optimizer = torch.optim.AdamW(
            list(self.critic_backbone.parameters())
            + list(self.value_head.parameters()),
            lr=config.critic_lr,
            eps=1e-5
        )

        self.kl_controller = AdaptiveKLController(
            config.kl_coef, config.target_kl, config.kl_horizon
        )

        self.running_rewards = deque(maxlen=100)
        self.global_step = 0
        os.makedirs(config.output_dir, exist_ok=True)

        if WANDB_AVAILABLE:
            wandb.init(
                project=config.wandb_project,
                name="ppo_rlhf",
                config=vars(config)
            )

    @torch.no_grad()
    def _generate_response(
        self, query_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate response and compute log probs.
        Returns response token ids and per-token log probs.
        """
        from generation.strategies import GenerationConfig, SamplingDecoder

        gen_config = GenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_k=0,
            top_p=1.0,
            do_sample=True,
        )
        decoder = SamplingDecoder(gen_config)
        self.actor.eval()

        full_ids = decoder.generate(self.actor, query_ids)
        response_ids = full_ids[:, query_ids.shape[1]:]
        self.actor.train()

        # Compute log probs of generated response
        log_probs = self._compute_log_probs(
            self.actor, full_ids, query_ids.shape[1]
        )
        return response_ids, log_probs

    def _compute_log_probs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        response_start: int
    ) -> torch.Tensor:
        """Compute per-token log probabilities of response tokens."""
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[:, response_start-1:-1, :]
        response_ids = input_ids[:, response_start:]

        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1,
            index=response_ids.unsqueeze(-1)
        ).squeeze(-1)
        return token_log_probs

    @torch.no_grad()
    def _compute_rewards(
        self,
        query_ids: torch.Tensor,
        response_ids: torch.Tensor,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute rewards with KL penalty.
        
        Total reward at last token:
            r_t = RM(x, y) - kl_coef * KL(pi || pi_ref)
        
        Per-token reward:
            r_t = -kl_coef * (log_pi(a_t|s_t) - log_pi_ref(a_t|s_t))
        Final token gets RM reward added.
        """
        full_ids = torch.cat([query_ids, response_ids], dim=1)

        # KL penalty per token: π_θ / π_ref
        kl = log_probs - ref_log_probs
        kl_penalty = -self.kl_controller.kl_coef * kl

        # RM score (scalar per sequence)
        rm_score = self.reward_model(full_ids)

        if self.config.normalize_rewards:
            rm_score = rm_score - rm_score.mean()
            rm_score = rm_score / (rm_score.std() + 1e-8)
            rm_score = rm_score.clamp(
                -self.config.reward_clip, self.config.reward_clip
            )

        # Add RM score to last token reward
        rewards = kl_penalty.clone()
        rewards[:, -1] += rm_score

        return rewards

    @torch.no_grad()
    def _compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generalized Advantage Estimation (GAE).
        
        A_t = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}
        where δ_t = r_t + γV(s_{t+1}) - V(s_t)
        
        Returns advantages and value targets (returns).
        """
        B, S = rewards.shape
        advantages = torch.zeros_like(rewards)
        last_gae = 0.0

        for t in reversed(range(S)):
            if t == S - 1:
                next_value = 0.0
            else:
                next_value = values[:, t + 1]

            delta = (
                rewards[:, t]
                + self.config.gamma * next_value
                - values[:, t]
            )
            last_gae = (
                delta
                + self.config.gamma * self.config.lam * last_gae
            )
            advantages[:, t] = last_gae

        returns = advantages + values
        return advantages, returns

    @torch.no_grad()
    def _compute_values(
        self,
        input_ids: torch.Tensor,
        response_start: int
    ) -> torch.Tensor:
        """Compute value estimates for response tokens."""
        outputs = self.critic_backbone(
            input_ids=input_ids,
            output_hidden_states=True
        )
        hidden = outputs.hidden_states[-1]
        response_hidden = hidden[:, response_start:, :]
        values = self.value_head(response_hidden)
        return values

    def _ppo_step(
        self,
        query_ids: torch.Tensor,
        response_ids: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor,
    ) -> dict:
        """
        Single PPO update step.
        
        Actor loss:
            L = -E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
        
        Critic loss:
            L = MSE(V(s), returns)
        """
        full_ids = torch.cat([query_ids, response_ids], dim=1)
        resp_start = query_ids.shape[1]

        # New log probs from actor
        new_log_probs = self._compute_log_probs(
            self.actor, full_ids, resp_start
        )

        # Align lengths
        min_len = min(new_log_probs.shape[1], old_log_probs.shape[1])
        new_log_probs = new_log_probs[:, :min_len]
        old_log_probs_t = old_log_probs[:, :min_len]
        advantages_t = advantages[:, :min_len]

        # Probability ratio
        ratio = torch.exp(new_log_probs - old_log_probs_t)

        # Normalize advantages
        adv_mean = advantages_t.mean()
        adv_std = advantages_t.std() + 1e-8
        norm_advantages = (advantages_t - adv_mean) / adv_std

        # PPO clipped objective
        pg_loss1 = -ratio * norm_advantages
        pg_loss2 = -(
            ratio.clamp(
                1 - self.config.clip_range,
                1 + self.config.clip_range
            ) * norm_advantages
        )
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Entropy bonus
        entropy = -(
            torch.exp(new_log_probs) * new_log_probs
        ).sum(-1).mean()

        # Critic / Value loss
        new_values = self._compute_values(full_ids, resp_start)
        min_val_len = min(new_values.shape[1], returns.shape[1])
        new_values = new_values[:, :min_val_len]
        returns_t = returns[:, :min_val_len]
        old_values_t = old_values[:, :min_val_len]

        # Clipped value loss
        value_clipped = old_values_t + (
            new_values - old_values_t
        ).clamp(-self.config.clip_range, self.config.clip_range)
        vf_loss = torch.max(
            F.mse_loss(new_values, returns_t),
            F.mse_loss(value_clipped, returns_t)
        )

        # Total loss
        total_loss = (
            pg_loss
            + self.config.vf_coef * vf_loss
            - self.config.ent_coef * entropy
        )

        # Update actor
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.config.max_grad_norm
        )
        nn.utils.clip_grad_norm_(
            list(self.critic_backbone.parameters())
            + list(self.value_head.parameters()),
            self.config.max_grad_norm
        )
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        # KL divergence
        with torch.no_grad():
            kl = (old_log_probs_t - new_log_probs).mean()

        return {
            "pg_loss": pg_loss.item(),
            "vf_loss": vf_loss.item(),
            "entropy": entropy.item(),
            "total_loss": total_loss.item(),
            "kl": kl.item(),
            "ratio_mean": ratio.mean().item(),
        }

    def train_step(
        self, queries: list[str]
    ) -> dict:
        """
        Full PPO training step for a batch of queries.
        
        1. Generate responses
        2. Score with reward model
        3. Compute advantages
        4. Run PPO_EPOCHS update iterations
        """
        config = self.config

        # Tokenize queries
        encoded = self.tokenizer.encode_batch(
            queries, padding=True, max_length=512
        )
        query_ids = torch.tensor(
            encoded["input_ids"], dtype=torch.long, device=self.device
        )

        # ── Rollout Phase ──────────────────────────────────────────
        with torch.no_grad():
            response_ids, log_probs = self._generate_response(query_ids)
            full_ids = torch.cat([query_ids, response_ids], dim=1)

            # Reference model log probs
            ref_log_probs = self._compute_log_probs(
                self.ref_model, full_ids, query_ids.shape[1]
            )

            # Compute rewards
            rewards = self._compute_rewards(
                query_ids, response_ids, log_probs, ref_log_probs
            )

            # Value estimates
            values = self._compute_values(
                full_ids, query_ids.shape[1]
            )

            # Align lengths
            min_len = min(
                rewards.shape[1],
                values.shape[1],
                log_probs.shape[1]
            )
            rewards = rewards[:, :min_len]
            values = values[:, :min_len]
            log_probs = log_probs[:, :min_len]

            # GAE advantages
            advantages, returns = self._compute_advantages(
                rewards, values
            )

        # Track rewards
        episode_reward = rewards.sum(dim=1).mean().item()
        self.running_rewards.append(episode_reward)

        # ── PPO Update Phase ────────────────────────────────────────
        all_metrics = []
        for ppo_epoch in range(config.ppo_epochs):
            metrics = self._ppo_step(
                query_ids, response_ids,
                log_probs, advantages, returns, values
            )
            all_metrics.append(metrics)

        avg_metrics = {
            k: sum(m[k] for m in all_metrics) / len(all_metrics)
            for k in all_metrics[0]
        }

        # Update KL controller
        self.kl_controller.update(
            avg_metrics["kl"], config.batch_size
        )

        avg_metrics["episode_reward"] = episode_reward
        avg_metrics["running_reward"] = (
            sum(self.running_rewards) / len(self.running_rewards)
        )
        avg_metrics["kl_coef"] = self.kl_controller.kl_coef

        self.global_step += 1
        return avg_metrics

    def train(self, prompt_dataset: list[str]):
        """Full PPO training loop."""
        config = self.config
        logger.info(
            f"Starting PPO RLHF | Steps: {config.total_steps}"
        )

        for step in range(config.total_steps):
            # Sample batch of prompts
            import random
            batch_prompts = random.sample(
                prompt_dataset,
                min(config.batch_size, len(prompt_dataset))
            )

            metrics = self.train_step(batch_prompts)

            if step % config.log_steps == 0:
                logger.info(
                    f"PPO Step {step}/{config.total_steps} | "
                    f"Reward: {metrics['episode_reward']:.4f} | "
                    f"PG Loss: {metrics['pg_loss']:.4f} | "
                    f"KL: {metrics['kl']:.4f} | "
                    f"KL Coef: {metrics['kl_coef']:.4f}"
                )
                if WANDB_AVAILABLE:
                    wandb.log(
                        {f"ppo/{k}": v for k, v in metrics.items()}
                    )

            if step % config.save_steps == 0:
                self._save(step)

        self._save("final")
        if WANDB_AVAILABLE:
            wandb.finish()

    def _save(self, tag):
        path = os.path.join(self.config.output_dir, str(tag))
        os.makedirs(path, exist_ok=True)
        actor = (
            self.actor.module
            if hasattr(self.actor, "module")
            else self.actor
        )
        if hasattr(actor, "save_pretrained"):
            actor.save_pretrained(path)
        else:
            torch.save(
                actor.state_dict(), os.path.join(path, "actor.pt")
            )
        torch.save(
            self.value_head.state_dict(),
            os.path.join(path, "value_head.pt")
        )
        logger.info(f"PPO checkpoint saved: {path}")