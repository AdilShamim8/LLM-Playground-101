"""
Human evaluation framework and leaderboard system.
Implements:
- ELO rating system (Chatbot Arena style)
- Pairwise comparison collection
- Leaderboard management
- A/B test framework
"""

import json
import math
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger


@dataclass
class ComparisonResult:
    comparison_id: str
    prompt: str
    model_a: str
    model_b: str
    response_a: str
    response_b: str
    winner: str             # "a" | "b" | "tie"
    judge_id: str
    timestamp: float
    metadata: dict = field(default_factory=dict)


@dataclass
class ModelRating:
    model_name: str
    elo: float = 1000.0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    total_comparisons: int = 0

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses + self.ties
        return self.wins / max(total, 1)

    @property
    def confidence_interval(self) -> tuple[float, float]:
        """95% CI using Wilson score interval."""
        n = self.total_comparisons
        if n == 0:
            return (0.0, 1.0)
        p = self.win_rate
        z = 1.96
        center = (p + z**2 / (2*n)) / (1 + z**2 / n)
        half_width = (
            z * math.sqrt(p*(1-p)/n + z**2/(4*n**2))
            / (1 + z**2/n)
        )
        return (
            max(0, center - half_width),
            min(1, center + half_width)
        )


class ELORatingSystem:
    """
    ELO rating system for chatbot evaluation.
    Used by Chatbot Arena (LMSYS).
    
    ELO update:
        E_a = 1 / (1 + 10^((R_b - R_a) / 400))
        R_a_new = R_a + K * (S_a - E_a)
    
    where S_a = 1 (win), 0.5 (tie), 0 (loss)
    K = 32 (sensitivity factor)
    """

    def __init__(
        self,
        k_factor: float = 32.0,
        initial_elo: float = 1000.0
    ):
        self.k_factor = k_factor
        self.initial_elo = initial_elo
        self.ratings: dict[str, ModelRating] = {}

    def _get_or_create(self, model_name: str) -> ModelRating:
        if model_name not in self.ratings:
            self.ratings[model_name] = ModelRating(
                model_name=model_name,
                elo=self.initial_elo
            )
        return self.ratings[model_name]

    def expected_score(self, elo_a: float, elo_b: float) -> float:
        """Expected score for player A against player B."""
        return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))

    def update(self, result: ComparisonResult):
        """Update ELO ratings based on comparison result."""
        model_a = self._get_or_create(result.model_a)
        model_b = self._get_or_create(result.model_b)

        elo_a = model_a.elo
        elo_b = model_b.elo

        expected_a = self.expected_score(elo_a, elo_b)
        expected_b = 1.0 - expected_a

        if result.winner == 'a':
            score_a, score_b = 1.0, 0.0
            model_a.wins += 1
            model_b.losses += 1
        elif result.winner == 'b':
            score_a, score_b = 0.0, 1.0
            model_b.wins += 1
            model_a.losses += 1
        else:  # tie
            score_a, score_b = 0.5, 0.5
            model_a.ties += 1
            model_b.ties += 1

        # ELO update
        model_a.elo += self.k_factor * (score_a - expected_a)
        model_b.elo += self.k_factor * (score_b - expected_b)
        model_a.total_comparisons += 1
        model_b.total_comparisons += 1

        logger.debug(
            f"ELO update: {result.model_a} "
            f"{elo_a:.1f} -> {model_a.elo:.1f} | "
            f"{result.model_b} "
            f"{elo_b:.1f} -> {model_b.elo:.1f}"
        )

    def get_leaderboard(self) -> list[dict]:
        """Return sorted leaderboard by ELO rating."""
        board = []
        for name, rating in self.ratings.items():
            ci_low, ci_high = rating.confidence_interval
            board.append({
                'rank': 0,
                'model': name,
                'elo': round(rating.elo, 1),
                'wins': rating.wins,
                'losses': rating.losses,
                'ties': rating.ties,
                'total': rating.total_comparisons,
                'win_rate': round(rating.win_rate * 100, 1),
                'ci_95_low': round(ci_low * 100, 1),
                'ci_95_high': round(ci_high * 100, 1),
            })

        board.sort(key=lambda x: x['elo'], reverse=True)
        for i, row in enumerate(board):
            row['rank'] = i + 1
        return board


class HumanEvalCollector:
    """
    Collects pairwise human preference judgments.
    Stores results and updates ELO leaderboard.
    """

    def __init__(
        self,
        storage_path: str = "./data/human_eval",
        elo_k: float = 32.0
    ):
        self.storage_path = storage_path
        self.elo_system = ELORatingSystem(k_factor=elo_k)
        self.comparisons: list[ComparisonResult] = []
        os.makedirs(storage_path, exist_ok=True)
        self._load_existing()

    def _load_existing(self):
        """Load saved comparisons and rebuild ELO."""
        fpath = os.path.join(self.storage_path, 'comparisons.jsonl')
        if not os.path.exists(fpath):
            return

        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    result = ComparisonResult(**data)
                    self.comparisons.append(result)
                    self.elo_system.update(result)
                except Exception as e:
                    logger.warning(f"Load error: {e}")

        logger.info(
            f"Loaded {len(self.comparisons)} comparisons"
        )

    def submit_comparison(
        self,
        prompt: str,
        model_a: str,
        response_a: str,
        model_b: str,
        response_b: str,
        winner: str,
        judge_id: str = "anonymous",
        metadata: Optional[dict] = None
    ) -> ComparisonResult:
        """Submit a human preference judgment."""
        assert winner in ('a', 'b', 'tie'), (
            "winner must be 'a', 'b', or 'tie'"
        )

        result = ComparisonResult(
            comparison_id=str(uuid.uuid4()),
            prompt=prompt,
            model_a=model_a,
            model_b=model_b,
            response_a=response_a,
            response_b=response_b,
            winner=winner,
            judge_id=judge_id,
            timestamp=time.time(),
            metadata=metadata or {}
        )

        self.comparisons.append(result)
        self.elo_system.update(result)
        self._save_comparison(result)

        logger.info(
            f"Comparison saved: {model_a} vs {model_b}, "
            f"winner={winner}"
        )
        return result

    def _save_comparison(self, result: ComparisonResult):
        fpath = os.path.join(self.storage_path, 'comparisons.jsonl')
        with open(fpath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(vars(result)) + '\n')

    def get_leaderboard(self) -> list[dict]:
        return self.elo_system.get_leaderboard()

    def save_leaderboard(self):
        """Save leaderboard snapshot to JSON."""
        board = self.get_leaderboard()
        fpath = os.path.join(self.storage_path, 'leaderboard.json')
        with open(fpath, 'w') as f:
            json.dump({
                'timestamp': time.time(),
                'leaderboard': board,
                'total_comparisons': len(self.comparisons)
            }, f, indent=2)
        logger.info(f"Leaderboard saved: {fpath}")
        return board

    def get_model_stats(self, model_name: str) -> Optional[dict]:
        """Get detailed stats for a specific model."""
        if model_name not in self.elo_system.ratings:
            return None
        rating = self.elo_system.ratings[model_name]
        ci = rating.confidence_interval
        return {
            'model': model_name,
            'elo': round(rating.elo, 1),
            'wins': rating.wins,
            'losses': rating.losses,
            'ties': rating.ties,
            'total': rating.total_comparisons,
            'win_rate': round(rating.win_rate * 100, 2),
            'ci_95': [round(ci[0]*100, 1), round(ci[1]*100, 1)],
        }

    def compute_agreement_rate(self) -> float:
        """
        Inter-annotator agreement rate.
        Measures consistency when same pair is judged multiple times.
        """
        pair_votes: dict[tuple, list[str]] = {}
        for comp in self.comparisons:
            key = tuple(sorted([comp.model_a, comp.model_b]))
            pair_votes.setdefault(key, []).append(comp.winner)

        agreements = 0
        total_pairs = 0
        for votes in pair_votes.values():
            if len(votes) < 2:
                continue
            most_common = max(set(votes), key=votes.count)
            agreements += votes.count(most_common) / len(votes)
            total_pairs += 1

        return agreements / max(total_pairs, 1)