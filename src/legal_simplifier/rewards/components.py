"""Composite reward: meaning (BERTScore) + simplicity (SARI, readability) - missing entities.

All scoring functions are batched. The BERTScore model is loaded once per
process and reused, because reloading ``roberta-large`` per call made the old
reward unusable inside an RL loop. Any object with a compatible ``score``
method can be passed instead, which is how the tests avoid downloading it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Protocol, cast

import textstat

from legal_simplifier.config import RewardConfig
from legal_simplifier.data.entities import check_entity_presence
from legal_simplifier.rewards.sari import sentence_sari


class SimilarityScorer(Protocol):
    def score(self, predictions: Sequence[str], references: Sequence[str]) -> list[float]: ...


class BertScoreScorer:
    """BERTScore-F1 with a model that is loaded once and scored in batches."""

    def __init__(
        self, model_type: str, *, rescale_with_baseline: bool = True, batch_size: int = 32
    ):
        from bert_score import BERTScorer

        self.batch_size = batch_size
        self._scorer = BERTScorer(
            model_type=model_type,
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


def readability_bonus(prediction: str, *, target_grade: float = 8.0) -> float:
    """Return a bonus in [0, 1] that grows as FKGL drops below ``target_grade``.

    Empty output scores 0. (The old version gave it the full bonus because
    textstat reports FKGL 0 for an empty string.)
    """

    if not prediction.strip():
        return 0.0
    fkgl = float(textstat.flesch_kincaid_grade(prediction))  # pyright: ignore[reportAttributeAccessIssue]
    return min(1.0, max(0.0, (target_grade - fkgl) / target_grade))


def entity_penalty(
    entities: Sequence[str], prediction: str, *, missing_penalty: float = 1.0
) -> float:
    """``missing_penalty`` times the fraction of ``entities`` missing from ``prediction``."""

    presence = check_entity_presence(entities, prediction)
    if not presence:
        return 0.0
    missing = sum(1 for kept in presence.values() if not kept)
    return missing_penalty * missing / len(presence)


@dataclass(frozen=True)
class RewardBreakdown:
    bertscore: float
    sari: float
    readability: float
    entity_penalty: float
    total: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def combine(
    *,
    bertscore: float,
    sari: float,
    readability: float,
    entity_penalty: float,
    config: RewardConfig,
) -> float:
    w = config.weights
    raw = (
        w.bertscore * bertscore
        + w.sari * sari
        + w.readability * readability
        - w.entity_penalty * entity_penalty
    )
    return min(config.clip_max, max(config.clip_min, raw))


def composite_rewards(
    *,
    clauses: Sequence[str],
    predictions: Sequence[str],
    references: Sequence[str],
    entities: Sequence[Sequence[str]],
    config: RewardConfig,
    scorer: SimilarityScorer | None = None,
) -> list[RewardBreakdown]:
    """Score a batch of predictions and return the per-component breakdown for each."""

    n = len(predictions)
    if not (len(clauses) == len(references) == len(entities) == n):
        raise ValueError("clauses, predictions, references and entities must have the same length")
    if scorer is None:
        scorer = get_bertscore_scorer(
            config.bertscore_model,
            config.bertscore_rescale_with_baseline,
            config.bertscore_batch_size,
        )
    bert = bertscore(predictions, references, scorer)
    sari_scores = sari(clauses, predictions, references)
    results: list[RewardBreakdown] = []
    for i in range(n):
        readability = readability_bonus(
            predictions[i], target_grade=config.readability_target_grade
        )
        penalty = entity_penalty(
            entities[i], predictions[i], missing_penalty=config.missing_entity_penalty
        )
        total = combine(
            bertscore=bert[i],
            sari=sari_scores[i],
            readability=readability,
            entity_penalty=penalty,
            config=config,
        )
        results.append(RewardBreakdown(bert[i], sari_scores[i], readability, penalty, total))
    return results
