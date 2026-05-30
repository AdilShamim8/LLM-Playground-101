"""
Comprehensive evaluation metrics.
Traditional NLP metrics + LLM-specific metrics.
"""

import math
import re
from collections import Counter
from typing import Optional

import torch
import torch.nn.functional as F
from loguru import logger


class PerplexityMetric:
    """
    Perplexity = exp(average negative log likelihood).
    Lower is better. Standard LM evaluation metric.
    """

    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def compute(
        self,
        texts: list[str],
        batch_size: int = 4,
        max_length: int = 1024,
        stride: int = 512
    ) -> dict:
        self.model.eval()
        all_nlls = []
        all_lengths = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]

            for text in batch:
                encoded = self.tokenizer.encode(
                    text, add_special_tokens=True
                )
                if len(encoded) < 2:
                    continue

                input_ids = torch.tensor(
                    [encoded], dtype=torch.long, device=self.device
                )
                seq_len = input_ids.shape[1]
                nlls = []

                for begin in range(0, seq_len, stride):
                    end = min(begin + max_length, seq_len)
                    trg_len = end - begin
                    chunk = input_ids[:, begin:end]
                    labels = chunk.clone()
                    if begin > 0:
                        labels[:, :stride] = -100

                    outputs = self.model(
                        input_ids=chunk, labels=labels
                    )
                    nlls.append(
                        outputs.loss.item() * trg_len
                    )
                    if end == seq_len:
                        break

                all_nlls.append(sum(nlls))
                all_lengths.append(seq_len)

        if not all_nlls:
            return {"perplexity": float("inf"), "nll": float("inf")}

        avg_nll = sum(all_nlls) / sum(all_lengths)
        perplexity = math.exp(min(avg_nll, 20))

        return {
            "perplexity": perplexity,
            "nll": avg_nll,
            "num_texts": len(all_nlls)
        }


class BLEUScore:
    """
    BLEU score for translation / generation quality.
    Precision-based n-gram overlap metric.
    """

    def __init__(self, max_n: int = 4):
        self.max_n = max_n

    def _get_ngrams(
        self, tokens: list[str], n: int
    ) -> Counter:
        return Counter(
            tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)
        )

    def _modified_precision(
        self,
        hypothesis: list[str],
        references: list[list[str]],
        n: int
    ) -> tuple[int, int]:
        hyp_ngrams = self._get_ngrams(hypothesis, n)
        clipped_count = 0
        total_count = sum(hyp_ngrams.values())

        for ngram, count in hyp_ngrams.items():
            max_ref = max(
                self._get_ngrams(ref, n)[ngram]
                for ref in references
            )
            clipped_count += min(count, max_ref)

        return clipped_count, total_count

    def _brevity_penalty(
        self, hyp_len: int, ref_lens: list[int]
    ) -> float:
        closest_ref_len = min(
            ref_lens, key=lambda l: abs(l - hyp_len)
        )
        if hyp_len >= closest_ref_len:
            return 1.0
        return math.exp(1 - closest_ref_len / hyp_len)

    def compute(
        self,
        hypotheses: list[str],
        references: list[list[str]]
    ) -> dict:
        assert len(hypotheses) == len(references)

        total_clips = [0] * self.max_n
        total_counts = [0] * self.max_n
        hyp_length = 0
        ref_length = 0

        for hyp, refs in zip(hypotheses, references):
            hyp_tokens = hyp.split()
            ref_tokens_list = [r.split() for r in refs]

            hyp_length += len(hyp_tokens)
            ref_length += min(
                len(r) for r in ref_tokens_list
            )

            for n in range(1, self.max_n + 1):
                clip, total = self._modified_precision(
                    hyp_tokens, ref_tokens_list, n
                )
                total_clips[n-1] += clip
                total_counts[n-1] += total

        precisions = []
        for n in range(self.max_n):
            if total_counts[n] == 0:
                precisions.append(0.0)
            else:
                p = total_clips[n] / total_counts[n]
                precisions.append(p if p > 0 else 1e-10)

        if all(p == 0 for p in precisions):
            bleu = 0.0
        else:
            log_avg = sum(
                math.log(p) for p in precisions if p > 0
            ) / self.max_n
            bp = self._brevity_penalty(hyp_length, [ref_length])
            bleu = bp * math.exp(log_avg)

        return {
            "bleu": bleu * 100,
            "bleu_1": precisions[0] * 100,
            "bleu_2": precisions[1] * 100,
            "bleu_3": precisions[2] * 100,
            "bleu_4": precisions[3] * 100,
            "brevity_penalty": self._brevity_penalty(
                hyp_length, [ref_length]
            )
        }


class ROUGEScore:
    """
    ROUGE scores for summarization evaluation.
    ROUGE-N: N-gram recall overlap.
    ROUGE-L: Longest Common Subsequence.
    """

    def _lcs_length(
        self, seq1: list, seq2: list
    ) -> int:
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

    def _rouge_n(
        self,
        hypothesis: str,
        reference: str,
        n: int = 2
    ) -> dict:
        hyp_tokens = hypothesis.split()
        ref_tokens = reference.split()

        def get_ngrams(tokens, n):
            return Counter(
                tuple(tokens[i:i+n])
                for i in range(len(tokens) - n + 1)
            )

        hyp_ngrams = get_ngrams(hyp_tokens, n)
        ref_ngrams = get_ngrams(ref_tokens, n)

        overlap = sum(
            min(hyp_ngrams[ng], ref_ngrams[ng])
            for ng in hyp_ngrams
        )
        precision = overlap / max(sum(hyp_ngrams.values()), 1)
        recall = overlap / max(sum(ref_ngrams.values()), 1)
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return {"precision": precision, "recall": recall, "f1": f1}

    def _rouge_l(
        self, hypothesis: str, reference: str
    ) -> dict:
        hyp = hypothesis.split()
        ref = reference.split()
        lcs = self._lcs_length(hyp, ref)
        precision = lcs / max(len(hyp), 1)
        recall = lcs / max(len(ref), 1)
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return {"precision": precision, "recall": recall, "f1": f1}

    def compute(
        self,
        hypotheses: list[str],
        references: list[str]
    ) -> dict:
        r1_scores, r2_scores, rl_scores = [], [], []

        for hyp, ref in zip(hypotheses, references):
            r1_scores.append(self._rouge_n(hyp, ref, n=1)["f1"])
            r2_scores.append(self._rouge_n(hyp, ref, n=2)["f1"])
            rl_scores.append(self._rouge_l(hyp, ref)["f1"])

        n = max(len(r1_scores), 1)
        return {
            "rouge1": sum(r1_scores) / n * 100,
            "rouge2": sum(r2_scores) / n * 100,
            "rougeL": sum(rl_scores) / n * 100,
        }


class EvaluationSuite:
    """Unified evaluation suite combining all metrics."""

    def __init__(self, model=None, tokenizer=None, device="cuda"):
        self.bleu = BLEUScore()
        self.rouge = ROUGEScore()
        self.ppl = (
            PerplexityMetric(model, tokenizer, device)
            if model and tokenizer else None
        )

    def evaluate_generation(
        self,
        hypotheses: list[str],
        references: list[str]
    ) -> dict:
        results = {}
        results.update(
            self.bleu.compute(hypotheses, [[r] for r in references])
        )
        results.update(
            self.rouge.compute(hypotheses, references)
        )
        return results

    def evaluate_lm(
        self,
        texts: list[str]
    ) -> dict:
        if self.ppl is None:
            return {}
        return self.ppl.compute(texts)