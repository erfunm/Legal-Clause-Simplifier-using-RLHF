"""Scoring building blocks: BERTScore, NLI entailment, SARI, reading grade, entities, format.

Model-based scorers are batched and loaded once per process (reloading
``roberta-large`` per call made the old reward unusable inside an RL loop).
Anything with a compatible method can be passed instead, which is how the
tests avoid downloads.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Protocol, cast

import textstat

from legal_simplifier.config import FormatConfig, SimplicityConfig
from legal_simplifier.data.entities import entity_recall
from legal_simplifier.rewards.hacking import (
    copy_overlap,
    detect_condition_drop,
    detect_modality_flip,
    detect_truncation,
    detect_word_salad,
    tokens,
)
from legal_simplifier.rewards.sari import sentence_sari


class SimilarityScorer(Protocol):
    def score(self, predictions: Sequence[str], references: Sequence[str]) -> list[float]: ...


class EntailmentScorer(Protocol):
    def entailment(self, premises: Sequence[str], hypotheses: Sequence[str]) -> list[float]: ...


class BertScoreScorer:
    """BERTScore-F1 with a model that is loaded once and scored in batches."""

    def __init__(
        self,
        model_type: str,
        *,
        rescale_with_baseline: bool = True,
        batch_size: int = 32,
        num_layers: int | None = None,
    ):
        from bert_score import BERTScorer

        self.batch_size = batch_size
        self._scorer = BERTScorer(
            model_type=model_type,
            num_layers=num_layers,  # None: bert-score's tuned layer for known models
            lang="en",
            rescale_with_baseline=rescale_with_baseline,
            batch_size=batch_size,
        )

    def score(self, predictions: Sequence[str], references: Sequence[str]) -> list[float]:
        if not predictions:
            return []
        # bert-score's return annotation is too loose for the type checker: it is (P, R, F1) tensors.
        _, _, f1 = cast(
            tuple[Any, Any, Any],
            self._scorer.score(list(predictions), list(references), batch_size=self.batch_size),
        )
        return [float(x) for x in f1.tolist()]


@lru_cache(maxsize=2)
def get_bertscore_scorer(
    model_type: str, rescale_with_baseline: bool = True, batch_size: int = 32
) -> BertScoreScorer:
    return BertScoreScorer(
        model_type, rescale_with_baseline=rescale_with_baseline, batch_size=batch_size
    )


class NLIScorer:
    """P(entailment) from an NLI cross-encoder, premise = clause, hypothesis = output.

    Long clauses are truncated from the premise only (``only_first``), so the
    output being judged is always seen in full.
    """

    def __init__(
        self,
        model: str,
        *,
        revision: str | None = None,
        entailment_label: str | None = None,
        max_length: int = 512,
        batch_size: int = 32,
        device: str | None = None,
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(model, revision=revision)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()  # pyright: ignore[reportArgumentType]
        self.max_length = max_length
        self.batch_size = batch_size
        labels = {str(v).lower(): int(k) for k, v in self.model.config.id2label.items()}
        if entailment_label is not None:
            if entailment_label.lower() not in labels:
                raise ValueError(
                    f"{model} has no label {entailment_label!r}; labels: {sorted(labels)}"
                )
            self.entailment_index = labels[entailment_label.lower()]
        else:
            matches = [i for name, i in labels.items() if "entail" in name]
            if len(matches) != 1:
                raise ValueError(
                    f"cannot find the entailment label of {model} ({sorted(labels)}); "
                    "set meaning.entailment_label"
                )
            self.entailment_index = matches[0]

    def entailment(self, premises: Sequence[str], hypotheses: Sequence[str]) -> list[float]:
        import torch

        if len(premises) != len(hypotheses):
            raise ValueError("premises and hypotheses must have the same length")
        scores: list[float] = []
        for start in range(0, len(premises), self.batch_size):
            batch = self.tokenizer(
                list(premises[start : start + self.batch_size]),
                list(hypotheses[start : start + self.batch_size]),
                truncation="only_first",
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                probs = self.model(**batch).logits.float().softmax(dim=-1)
            scores.extend(probs[:, self.entailment_index].tolist())
        return scores


@lru_cache(maxsize=2)
def get_nli_scorer(
    model: str,
    revision: str | None = None,
    entailment_label: str | None = None,
    max_length: int = 512,
    batch_size: int = 32,
) -> NLIScorer:
    return NLIScorer(
        model,
        revision=revision,
        entailment_label=entailment_label,
        max_length=max_length,
        batch_size=batch_size,
    )


# Split before a condition/exception that follows a comma, semicolon or colon, keeping
# the marker with its part: "..., unless X" -> ["...", "unless X"].
_PART_BREAK = re.compile(
    r"(?<=[,;:])\s+(?=(?:unless|except|provided|subject\s+to|if|only\s+if|until|"
    r"notwithstanding|in\s+the\s+event|so\s+long\s+as|as\s+long\s+as|but)\b)",
    re.IGNORECASE,
)
_SENTENCE_BREAK = re.compile(r"(?<=[.!?;])\s+")


def clause_parts(clause: str, *, max_parts: int = 8, min_words: int = 3) -> list[str]:
    """Sentences and conditional parts of a clause, for sentence-level reverse NLI.

    Very short pieces are merged into the previous part, and anything past
    ``max_parts`` is merged into the last one, so the number of NLI calls per
    clause is bounded.
    """

    pieces = [
        p.strip(" ,;:")
        for sentence in _SENTENCE_BREAK.split(" ".join(clause.split()))
        for p in _PART_BREAK.split(sentence)
    ]
    parts: list[str] = []
    for piece in pieces:
        if not piece:
            continue
        if parts and (len(piece.split()) < min_words or len(parts) >= max_parts):
            parts[-1] = f"{parts[-1]} {piece}"
        else:
            parts.append(piece)
    if len(parts) > 1 and len(parts[0].split()) < min_words:
        parts[1] = f"{parts[0]} {parts[1]}"
        parts.pop(0)
    return parts or [clause.strip()]


def coverage_entailment(
    clauses: Sequence[str],
    outputs: Sequence[str],
    scorer: EntailmentScorer,
    *,
    max_parts: int = 8,
    min_words: int = 3,
) -> list[float]:
    """Mean P(entailment | output -> part) over each clause's parts, in one batched call."""

    premises: list[str] = []
    hypotheses: list[str] = []
    owner: list[int] = []
    for i, (clause, output) in enumerate(zip(clauses, outputs, strict=True)):
        for part in clause_parts(clause, max_parts=max_parts, min_words=min_words):
            premises.append(output)
            hypotheses.append(part)
            owner.append(i)
    values = scorer.entailment(premises, hypotheses) if premises else []
    totals = [0.0] * len(clauses)
    counts = [0] * len(clauses)
    for i, value in zip(owner, values, strict=True):
        totals[i] += float(value)
        counts[i] += 1
    return [t / c if c else 0.0 for t, c in zip(totals, counts, strict=True)]


def bertscore(
    predictions: Sequence[str], references: Sequence[str], scorer: SimilarityScorer
) -> list[float]:
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    return scorer.score(predictions, references)


def sari(
    sources: Sequence[str], predictions: Sequence[str], references: Sequence[str]
) -> list[float]:
    return [
        sentence_sari(s, p, [r]) for s, p, r in zip(sources, predictions, references, strict=True)
    ]


def fkgl(text: str) -> float:
    return float(textstat.flesch_kincaid_grade(text))  # pyright: ignore[reportAttributeAccessIssue]


def fkgl_band_score(prediction: str, config: SimplicityConfig) -> float:
    """1 inside [fkgl_low, fkgl_high], falling linearly to 0 over ``fkgl_margin`` either side.

    The lower edge matters: without it, the cheapest way to a low grade is a
    string of two-word sentences. Empty output scores 0 (textstat reports FKGL
    0 for an empty string, which the old bonus rewarded).
    """

    if not prediction.strip():
        return 0.0
    grade = fkgl(prediction)
    if grade < config.fkgl_low:
        distance = config.fkgl_low - grade
    elif grade > config.fkgl_high:
        distance = grade - config.fkgl_high
    else:
        return 1.0
    return max(0.0, 1.0 - distance / config.fkgl_margin)


def simplicity_score(
    clause: str, prediction: str, reference: str, config: SimplicityConfig
) -> float:
    if config.method == "sari":
        return sentence_sari(clause, prediction, [reference])
    band = fkgl_band_score(prediction, config)
    if config.method == "fkgl_band":
        return band
    return (sentence_sari(clause, prediction, [reference]) + band) / 2


def entity_preservation_score(entities: Sequence[str], prediction: str) -> float:
    """Fraction of required entities present after normalisation; 1 when there are none."""

    return entity_recall(entities, prediction)


def format_score(
    clause: str, prediction: str, entities: Sequence[str], config: FormatConfig
) -> tuple[float, list[str]]:
    """A penalty in [-1, 0] and the names of the problems found (empty when clean)."""

    if not prediction.strip():
        return -1.0, ["empty"]
    penalty = 0.0
    problems: list[str] = []
    if copy_overlap(clause, prediction) >= config.copy_threshold:
        penalty += config.copy_penalty
        problems.append("verbatim_copy")
    if len(tokens(prediction)) > config.max_length_ratio * max(1, len(tokens(clause))):
        penalty += config.runaway_penalty
        problems.append("runaway_length")
    if detect_truncation(clause, prediction).flagged:
        penalty += config.truncation_penalty
        problems.append("truncation")
    if detect_word_salad(prediction, entities).flagged:
        penalty += config.word_salad_penalty
        problems.append("word_salad")
    if detect_modality_flip(clause, prediction).flagged:
        penalty += config.modality_flip_penalty
        problems.append("modality_flip")
    if detect_condition_drop(clause, prediction).flagged:
        penalty += config.condition_drop_penalty
        problems.append("condition_drop")
    return max(-1.0, -penalty), problems
