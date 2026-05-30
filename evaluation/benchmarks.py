"""
Task-specific benchmark evaluations.
Implements: HellaSwag, MMLU, GSM8K, HumanEval-style tasks.
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from loguru import logger


@dataclass
class BenchmarkResult:
    task: str
    accuracy: float
    num_examples: int
    details: dict


class MultipleChoiceBenchmark:
    """
    Multiple choice evaluation (HellaSwag, MMLU, ARC, etc.)
    Uses log-likelihood scoring of each answer choice.
    """

    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def _score_continuation(
        self,
        context_ids: list[int],
        continuation_ids: list[int]
    ) -> float:
        """Score the log-probability of a continuation given context."""
        input_ids = torch.tensor(
            [context_ids + continuation_ids],
            dtype=torch.long,
            device=self.device
        )
        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits[0]

        # Only score continuation tokens
        cont_start = len(context_ids)
        cont_logits = logits[cont_start-1:-1]
        cont_ids = torch.tensor(
            continuation_ids, dtype=torch.long, device=self.device
        )

        log_probs = F.log_softmax(cont_logits, dim=-1)
        scores = log_probs[
            torch.arange(len(continuation_ids)), cont_ids
        ]
        # Normalize by length
        return scores.mean().item()

    def evaluate(
        self,
        examples: list[dict],
        task_name: str = "multiple_choice"
    ) -> BenchmarkResult:
        """
        Evaluate on multiple choice examples.
        
        Each example format:
        {
            "context": "The sky is...",
            "choices": ["blue", "green", "red", "purple"],
            "gold": 0  # index of correct answer
        }
        """
        self.model.eval()
        correct = 0
        total = 0

        for ex in examples:
            context = ex["context"]
            choices = ex["choices"]
            gold = ex["gold"]

            context_ids = self.tokenizer.encode(
                context, add_special_tokens=True
            )

            scores = []
            for choice in choices:
                choice_ids = self.tokenizer.encode(
                    " " + choice, add_special_tokens=False
                )
                score = self._score_continuation(
                    context_ids, choice_ids
                )
                scores.append(score)

            predicted = scores.index(max(scores))
            if predicted == gold:
                correct += 1
            total += 1

        accuracy = correct / max(total, 1)
        logger.info(
            f"{task_name}: {accuracy*100:.2f}% "
            f"({correct}/{total})"
        )

        return BenchmarkResult(
            task=task_name,
            accuracy=accuracy,
            num_examples=total,
            details={"correct": correct, "total": total}
        )


class MMLUBenchmark:
    """
    Massive Multitask Language Understanding.
    57 academic subjects, multiple choice.
    """

    SUBJECTS = [
        "abstract_algebra", "anatomy", "astronomy",
        "business_ethics", "clinical_knowledge",
        "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics",
        "college_medicine", "college_physics",
        "computer_security", "conceptual_physics",
        "econometrics", "electrical_engineering",
        "elementary_mathematics", "formal_logic",
        "global_facts", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science",
        "high_school_european_history", "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_mathematics",
        "high_school_microeconomics",
        "high_school_physics",
        "high_school_psychology",
        "high_school_statistics",
        "high_school_us_history",
        "high_school_world_history",
        "human_aging",
        "human_sexuality",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "machine_learning",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "moral_disputes",
        "moral_scenarios",
        "nutrition",
        "philosophy",
        "prehistory",
        "professional_accounting",
        "professional_law",
        "professional_medicine",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
        "virology",
        "world_religions"
    ]

    def __init__(self, model, tokenizer, device="cuda"):
        self.mc_bench = MultipleChoiceBenchmark(
            model, tokenizer, device
        )
        self.model = model
        self.tokenizer = tokenizer

    def _format_example(self, ex: dict, subject: str) -> dict:
        """Format MMLU example for evaluation."""
        choices = ["A", "B", "C", "D"]
        question = ex.get("question", "")
        options = ex.get("choices", [])
        answer = ex.get("answer", 0)

        context = (
            f"The following is a multiple choice question about "
            f"{subject.replace('_', ' ')}.\n\n"
            f"Question: {question}\n"
        )
        for i, (choice, opt) in enumerate(zip(choices, options)):
            context += f"{choice}. {opt}\n"
        context += "Answer:"

        return {
            "context": context,
            "choices": choices[:len(options)],
            "gold": answer if isinstance(answer, int) else (
                choices.index(answer) if answer in choices else 0
            )
        }

    def evaluate(
        self,
        data_dir: str,
        subjects: Optional[list[str]] = None,
        num_few_shot: int = 5
    ) -> dict:
        if subjects is None:
            subjects = self.SUBJECTS

        results = {}
        all_acc = []

        for subject in subjects:
            filepath = os.path.join(
                data_dir, f"{subject}_test.jsonl"
            )
            if not os.path.exists(filepath):
                logger.warning(f"MMLU data not found: {filepath}")
                continue

            examples = []
            with open(filepath, encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        examples.append(
                            self._format_example(item, subject)
                        )
                    except Exception:
                        continue

            if not examples:
                continue

            result = self.mc_bench.evaluate(examples, subject)
            results[subject] = result.accuracy
            all_acc.append(result.accuracy)

        macro_avg = sum(all_acc) / max(len(all_acc), 1)
        logger.info(f"MMLU Macro Average: {macro_avg*100:.2f}%")

        return {
            "mmlu_avg": macro_avg,
            "per_subject": results,
            "num_subjects": len(results)
        }


class GSM8KBenchmark:
    """
    Grade School Math 8K benchmark.
    Tests mathematical reasoning with chain-of-thought.
    """

    def __init__(self, model, tokenizer, generator, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.generator = generator
        self.device = device

    def _extract_answer(self, text: str) -> Optional[str]:
        patterns = [
            r"####\s*([-+]?\d+\.?\d*)",
            r"(?:answer is|=)\s*([-+]?\d+\.?\d*)",
            r"\\boxed\{([-+]?\d+\.?\d*)\}",
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        nums = re.findall(r"[-+]?\d+\.?\d*", text)
        return nums[-1] if nums else None

    def evaluate(
        self,
        examples: list[dict],
        num_few_shot: int = 8
    ) -> BenchmarkResult:
        few_shot_prompt = self._build_few_shot_prompt(
            examples[:num_few_shot]
        )
        test_examples = examples[num_few_shot:]

        correct = 0
        total = 0

        for ex in test_examples[:100]:  # Quick eval
            question = ex.get("question", "")
            gold_answer = self._extract_answer(
                ex.get("answer", "")
            )

            prompt = (
                few_shot_prompt
                + f"Question: {question}\nAnswer: "
            )

            response = self.generator.generate(
                prompt,
                strategy="greedy",
                max_new_tokens=200
            )

            pred_answer = self._extract_answer(response)

            if (
                pred_answer and gold_answer
                and pred_answer.replace(",", "")
                == gold_answer.replace(",", "")
            ):
                correct += 1
            total += 1

        accuracy = correct / max(total, 1)
        logger.info(
            f"GSM8K: {accuracy*100:.2f}% ({correct}/{total})"
        )

        return BenchmarkResult(
            task="gsm8k",
            accuracy=accuracy,
            num_examples=total,
            details={"correct": correct}
        )

    def _build_few_shot_prompt(
        self, examples: list[dict]
    ) -> str:
        prompt = ""
        for ex in examples[:3]:
            q = ex.get("question", "")
            a = ex.get("answer", "")
            prompt += f"Question: {q}\nAnswer: {a}\n\n"
        return prompt