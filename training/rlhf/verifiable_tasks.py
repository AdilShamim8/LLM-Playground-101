"""
Verifiable reward tasks for RLHF without a reward model.
Used in DeepSeek-R1, process reward models.
Provides ground-truth reward signals for:
- Math problems
- Code execution
- Factual Q&A
- Format compliance
"""

import ast
import re
import subprocess
import tempfile
import os
import sys
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from loguru import logger


@dataclass
class VerifiableReward:
    reward: float
    passed: bool
    reason: str
    details: dict


class VerifiableTask(ABC):
    """Abstract base for verifiable reward tasks."""

    @abstractmethod
    def compute_reward(
        self, prompt: str, response: str, ground_truth: str
    ) -> VerifiableReward:
        pass

    @abstractmethod
    def extract_answer(self, response: str) -> Optional[str]:
        pass


class MathVerifier(VerifiableTask):
    """
    Verifiable rewards for math problems.
    Extracts and compares numeric answers.
    Used by: DeepSeek-R1, Qwen-Math.
    """

    ANSWER_PATTERNS = [
        r"(?:the answer is|answer:|=)\s*([-+]?\d+\.?\d*)",
        r"\\boxed\{([-+]?\d+\.?\d*)\}",
        r"(?:therefore|thus|so),?\s*([-+]?\d+\.?\d*)\s*$",
        r"=\s*([-+]?\d+\.?\d*)\s*$",
    ]

    def extract_answer(self, response: str) -> Optional[str]:
        """Extract numeric answer from model response."""
        for pattern in self.ANSWER_PATTERNS:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Last number in response
        numbers = re.findall(r"[-+]?\d+\.?\d*", response)
        return numbers[-1] if numbers else None

    def _parse_number(self, s: str) -> Optional[float]:
        try:
            return float(s.replace(",", ""))
        except (ValueError, TypeError):
            return None

    def compute_reward(
        self,
        prompt: str,
        response: str,
        ground_truth: str
    ) -> VerifiableReward:
        predicted = self.extract_answer(response)
        if predicted is None:
            return VerifiableReward(
                reward=0.0,
                passed=False,
                reason="No answer extracted",
                details={}
            )

        pred_val = self._parse_number(predicted)
        true_val = self._parse_number(ground_truth)

        if pred_val is None or true_val is None:
            return VerifiableReward(
                reward=0.0,
                passed=False,
                reason="Failed to parse numbers",
                details={"predicted": predicted, "truth": ground_truth}
            )

        # Exact match with tolerance
        if abs(true_val) < 1e-6:
            correct = abs(pred_val - true_val) < 1e-6
        else:
            correct = abs(pred_val - true_val) / abs(true_val) < 0.01

        # Partial rewards
        if correct:
            reward = 1.0
        else:
            # Penalize wrong answer
            error_ratio = abs(pred_val - true_val) / (abs(true_val) + 1e-8)
            reward = max(-0.5, -0.5 * min(error_ratio, 1.0))

        # Format bonus: reward for showing work
        has_steps = len(response.split("\n")) > 3
        if correct and has_steps:
            reward = min(1.0, reward + 0.1)

        return VerifiableReward(
            reward=reward,
            passed=correct,
            reason="correct" if correct else "incorrect",
            details={
                "predicted": pred_val,
                "ground_truth": true_val,
                "error_ratio": abs(pred_val - true_val) / (
                    abs(true_val) + 1e-8
                )
            }
        )


class CodeVerifier(VerifiableTask):
    """
    Verifiable rewards for code generation.
    Executes generated code in sandbox and checks output.
    """

    def __init__(
        self,
        timeout: int = 10,
        max_output_len: int = 1000
    ):
        self.timeout = timeout
        self.max_output_len = max_output_len

    def extract_answer(self, response: str) -> Optional[str]:
        """Extract Python code block from response."""
        patterns = [
            r"```python\n(.*?)```",
            r"```\n(.*?)```",
            r"`(.*?)`",
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                return match.group(1).strip()
        return response.strip()

    def _run_code(
        self, code: str, test_input: str = ""
    ) -> tuple[bool, str, str]:
        """Execute code in subprocess sandbox."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py",
            delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmpfile = f.name

        try:
            python_exe = sys.executable or "python"
            result = subprocess.run(
                [python_exe, tmpfile],
                input=test_input,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
            )
            stdout = result.stdout[:self.max_output_len]
            stderr = result.stderr[:500]
            return result.returncode == 0, stdout, stderr

        except subprocess.TimeoutExpired:
            return False, "", "Timeout"
        except Exception as e:
            return False, "", str(e)
        finally:
            try:
                os.unlink(tmpfile)
            except Exception:
                pass

    def compute_reward(
        self,
        prompt: str,
        response: str,
        ground_truth: str
    ) -> VerifiableReward:
        code = self.extract_answer(response)
        if not code:
            return VerifiableReward(
                reward=-1.0,
                passed=False,
                reason="No code extracted",
                details={}
            )

        # Syntax check
        try:
            ast.parse(code)
        except SyntaxError as e:
            return VerifiableReward(
                reward=-0.5,
                passed=False,
                reason=f"Syntax error: {e}",
                details={"code": code[:200]}
            )

        # Execute and check output
        success, stdout, stderr = self._run_code(code)
        if not success:
            return VerifiableReward(
                reward=-0.3,
                passed=False,
                reason=f"Runtime error: {stderr[:100]}",
                details={"stderr": stderr}
            )

        # Compare output with expected
        expected = ground_truth.strip()
        actual = stdout.strip()

        if actual == expected:
            reward = 1.0
            passed = True
            reason = "Output matches"
        elif expected in actual or actual in expected:
            reward = 0.5
            passed = False
            reason = "Partial output match"
        else:
            reward = 0.1
            passed = False
            reason = f"Wrong output: got '{actual[:50]}'"

        return VerifiableReward(
            reward=reward,
            passed=passed,
            reason=reason,
            details={
                "actual": actual[:200],
                "expected": expected[:200]
            }
        )


class FormatVerifier(VerifiableTask):
    """
    Verifiable rewards for format compliance.
    Checks JSON, XML, specific structure requirements.
    """

    def extract_answer(self, response: str) -> Optional[str]:
        return response.strip()

    def compute_reward(
        self,
        prompt: str,
        response: str,
        ground_truth: str
    ) -> VerifiableReward:
        import json

        format_type = ground_truth.lower()
        score = 0.0
        reasons = []

        if format_type == "json":
            try:
                json.loads(response)
                score = 1.0
                reasons.append("Valid JSON")
            except json.JSONDecodeError as e:
                score = 0.0
                reasons.append(f"Invalid JSON: {e}")

        elif format_type == "bullet_list":
            lines = response.strip().split("\n")
            bullet_lines = [
                l for l in lines
                if l.strip().startswith(("-", "•", "*", "·"))
            ]
            ratio = len(bullet_lines) / max(len(lines), 1)
            score = min(1.0, ratio * 2)
            reasons.append(f"Bullet ratio: {ratio:.2f}")

        elif format_type == "concise":
            words = len(response.split())
            if words <= 50:
                score = 1.0
            elif words <= 100:
                score = 0.5
            else:
                score = max(0.0, 1.0 - (words - 100) / 200)
            reasons.append(f"Word count: {words}")

        return VerifiableReward(
            reward=score,
            passed=score >= 0.8,
            reason="; ".join(reasons),
            details={"format": format_type, "score": score}
        )


class VerifiableRewardRouter:
    """
    Routes prompts to appropriate verifier based on task type.
    Combines multiple reward signals.
    """

    def __init__(self):
        self.verifiers = {
            "math": MathVerifier(),
            "code": CodeVerifier(),
            "format": FormatVerifier(),
        }

    def compute_reward(
        self,
        prompt: str,
        response: str,
        ground_truth: str,
        task_type: str = "math"
    ) -> VerifiableReward:
        verifier = self.verifiers.get(task_type)
        if not verifier:
            logger.warning(f"Unknown task type: {task_type}")
            return VerifiableReward(
                reward=0.0, passed=False,
                reason="Unknown task", details={}
            )
        return verifier.compute_reward(prompt, response, ground_truth)