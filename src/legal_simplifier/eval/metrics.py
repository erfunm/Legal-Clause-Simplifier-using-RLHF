"""Per-example automatic metrics, built from the reward code (nothing reimplemented).

=====================  =======================================================
``sari``               :func:`rewards.components.sari` against the reference
``bertscore``          :func:`rewards.components.bertscore` F1 against the reference
``fkgl``               :func:`rewards.components.fkgl` of the output (``None`` if empty)
``entity_recall``      :func:`rewards.components.entity_preservation_score`
``nli_faithfulness``   ``RewardSuite.meaning`` with ``nli``: P(entail | clause -> output)
``nli_coverage``       ``RewardSuite.meaning`` with ``nli_coverage``: each clause part <- output
``compression_ratio``  output words / clause words
``hallucinated``       ``nli_faithfulness`` below ``metrics.faithfulness_threshold``
``detectors``          every :mod:`rewards.hacking` detector (``condition_drop`` included)
=====================  =======================================================

The NLI model is loaded once and shared by the faithfulness and coverage scores.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from legal_simplifier.config import EvalMetricsConfig, RewardConfig
from legal_simplifier.eval.items import EvalItem
from legal_simplifier.rewards.components import (
    EntailmentScorer,
    SimilarityScorer,
    entity_preservation_score,
    fkgl,
    get_bertscore_scorer,
    get_nli_scorer,
    sari,
)
from legal_simplifier.rewards.hacking import DETECTORS, run_detectors, tokens
from legal_simplifier.rewards.suite import RewardSuite

METRICS = (
    "sari",
    "bertscore",
    "fkgl",
    "entity_recall",
    "nli_faithfulness",
    "nli_coverage",
    "compression_ratio",
    "hallucinated",
)
DETECTOR_METRICS = (*(f"detector/{d}" for d in DETECTORS), "detector/any")


class Scorers:
    """Model-based scorers, loaded lazily; tests pass fakes instead."""

    def __init__(
        self,
        config: EvalMetricsConfig,
        *,
        entailment: EntailmentScorer | None = None,
        similarity: SimilarityScorer | None = None,
    ):
        self.config = config
        self._entailment = entailment
        self._similarity = similarity

    def entailment(self) -> EntailmentScorer:
        if self._entailment is None:
            m = self.config.meaning
            self._entailment = get_nli_scorer(
                m.nli_model, m.nli_revision, m.entailment_label, m.max_length, m.batch_size
            )
        return self._entailment

    def similarity(self) -> SimilarityScorer | None:
        if self.config.bertscore_model is None:
            return None
        if self._similarity is None:
            from legal_simplifier.rewards.components import BertScoreScorer

            if self.config.bertscore_num_layers is not None:  # small/test models need it
                self._similarity = BertScoreScorer(
                    self.config.bertscore_model,
                    rescale_with_baseline=self.config.bertscore_rescale_with_baseline,
                    batch_size=self.config.bertscore_batch_size,
                    num_layers=self.config.bertscore_num_layers,
                )
            else:
                self._similarity = get_bertscore_scorer(
                    self.config.bertscore_model,
                    self.config.bertscore_rescale_with_baseline,
                    self.config.bertscore_batch_size,
                )
        return self._similarity

    def suite(self, method: str) -> RewardSuite:
        meaning = self.config.meaning.model_copy(update={"method": method})
        return RewardSuite(RewardConfig(meaning=meaning), entailment=self.entailment())


def score_outputs(
    items: Sequence[EvalItem], outputs: Sequence[str], scorers: Scorers
) -> list[dict[str, Any]]:
    """One dict per item: the metrics above plus ``detectors`` and their details."""

    if len(items) != len(outputs):
        raise ValueError("one output per item is required")
    clauses = [i.clause for i in items]
    references = [i.reference for i in items]
    sari_scores = sari(clauses, outputs, references)
    similarity = scorers.similarity()
    bert: list[float | None] = [None] * len(items)  # None: BERTScore disabled
    if similarity is not None:
        bert = [0.0] * len(items)  # an empty output scores 0
        keep = [k for k, o in enumerate(outputs) if o.strip()]
        if keep:
            values = similarity.score([outputs[k] for k in keep], [references[k] for k in keep])
            for k, v in zip(keep, values, strict=True):
                bert[k] = float(v)
    faithful = scorers.suite("nli").meaning(clauses, outputs, references)
    coverage = scorers.suite("nli_coverage").meaning(clauses, outputs, references)
    threshold = scorers.config.faithfulness_threshold

    rows = []
    for k, (item, output) in enumerate(zip(items, outputs, strict=True)):
        findings = run_detectors(item.clause, output, item.entities)
        fired = {name: f.flagged for name, f in findings.items()}
        rows.append(
            {
                "sari": sari_scores[k],
                "bertscore": bert[k],
                "fkgl": fkgl(output) if output.strip() else None,
                "entity_recall": entity_preservation_score(item.entities, output),
                "nli_faithfulness": faithful[k],
                "nli_coverage": coverage[k],
                "compression_ratio": len(tokens(output)) / max(1, len(tokens(item.clause))),
                "hallucinated": float(faithful[k] < threshold),
                "empty": not output.strip(),
                "detectors": fired,
                "detector_details": {n: f.detail for n, f in findings.items() if f.flagged},
            }
        )
    return rows


def flat_metrics(row: dict[str, Any]) -> dict[str, float | None]:
    """The numeric view of one scored example, detectors as 0/1 under ``detector/<name>``."""

    metrics = row["metrics"]
    flat: dict[str, float | None] = {m: metrics.get(m) for m in METRICS}
    detectors = metrics.get("detectors", {})
    for name in DETECTORS:
        flat[f"detector/{name}"] = float(bool(detectors.get(name)))
    flat["detector/any"] = float(any(detectors.values()))
    for criterion, value in (row.get("judge") or {}).items():
        flat[f"judge/{criterion}"] = value
    return flat
