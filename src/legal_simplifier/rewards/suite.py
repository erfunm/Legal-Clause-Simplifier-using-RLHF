"""The four reward functions, as GRPO reward callables and as one weighted composite.

* ``meaning``: NLI P(entailment | clause -> output) by default, or BERTScore-F1
  against the reference (``meaning.method``). Range [0, 1] for NLI.
* ``simplicity``: SARI against the reference, an FKGL target band, or their
  mean. Range [0, 1].
* ``entity_preservation``: fraction of required entities kept, after
  normalising formatting ("$1,000,000" = "one million dollars"). Range [0, 1].
* ``format``: 0 when clean, down to -1 for empty output, verbatim copying,
  runaway length and the reward-hacking detectors. Range [-1, 0].

GRPO receives the four as separate functions with ``reward_weights``, so TRL
logs each one as ``rewards/<name>/mean``. DPO pair ranking and evaluation use
:meth:`RewardSuite.score`, the same functions combined with the same weights,
so all three stages agree on what "better" means.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from legal_simplifier.config import RewardConfig
from legal_simplifier.rewards.components import (
    EntailmentScorer,
    SimilarityScorer,
    entity_preservation_score,
    format_score,
    get_bertscore_scorer,
    get_nli_scorer,
    simplicity_score,
)

REWARD_NAMES = ("meaning", "simplicity", "entity_preservation", "format")
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def completion_text(completion: Any) -> str:
    """Text of a TRL completion (a string, or a list of chat messages), think block removed."""

    if isinstance(completion, list):
        completion = "".join(
            m.get("content", "") for m in completion if m.get("role", "assistant") == "assistant"
        )
    return " ".join(_THINK.sub("", str(completion)).split())


@dataclass(frozen=True)
class RewardBreakdown:
    meaning: float
    simplicity: float
    entity_preservation: float
    format: float
    total: float
    problems: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["problems"] = list(self.problems)
        return data


class RewardSuite:
    """Scores outputs; model-based scorers are loaded lazily and only when needed."""

    def __init__(
        self,
        config: RewardConfig,
        *,
        entailment: EntailmentScorer | None = None,
        similarity: SimilarityScorer | None = None,
    ):
        self.config = config
        self._entailment = entailment
        self._similarity = similarity

    @property
    def weights(self) -> list[float]:
        w = self.config.weights
        return [w.meaning, w.simplicity, w.entity_preservation, w.format]

    def _nli(self) -> EntailmentScorer:
        if self._entailment is None:
            m = self.config.meaning
            self._entailment = get_nli_scorer(
                m.nli_model, m.nli_revision, m.entailment_label, m.max_length, m.batch_size
            )
        return self._entailment

    def _bertscore(self) -> SimilarityScorer:
        if self._similarity is None:
            m = self.config.meaning
            self._similarity = get_bertscore_scorer(
                m.bertscore_model, m.bertscore_rescale_with_baseline, m.batch_size
            )
        return self._similarity

    # -- the four functions, batched ------------------------------------------

    def meaning(
        self, clauses: Sequence[str], outputs: Sequence[str], references: Sequence[str]
    ) -> list[float]:
        scores = [0.0] * len(outputs)
        keep = [i for i, o in enumerate(outputs) if o.strip()]  # empty output means nothing
        if not keep:
            return scores
        if self.config.meaning.method == "nli":
            values = self._nli().entailment([clauses[i] for i in keep], [outputs[i] for i in keep])
        else:
            values = self._bertscore().score(
                [outputs[i] for i in keep], [references[i] for i in keep]
            )
        for i, value in zip(keep, values, strict=True):
            scores[i] = float(value)
        return scores

    def simplicity(
        self, clauses: Sequence[str], outputs: Sequence[str], references: Sequence[str]
    ) -> list[float]:
        cfg = self.config.simplicity
        return [
            simplicity_score(c, o, r, cfg)
            for c, o, r in zip(clauses, outputs, references, strict=True)
        ]

    def entity_preservation(
        self, outputs: Sequence[str], entities: Sequence[Sequence[str]]
    ) -> list[float]:
        return [entity_preservation_score(e, o) for o, e in zip(outputs, entities, strict=True)]

    def format(
        self, clauses: Sequence[str], outputs: Sequence[str], entities: Sequence[Sequence[str]]
    ) -> list[float]:
        cfg = self.config.format
        return [
            format_score(c, o, e, cfg)[0]
            for c, o, e in zip(clauses, outputs, entities, strict=True)
        ]

    # -- composite --------------------------------------------------------------

    def score(
        self,
        clauses: Sequence[str],
        outputs: Sequence[str],
        references: Sequence[str],
        entities: Sequence[Sequence[str]],
    ) -> list[RewardBreakdown]:
        n = len(outputs)
        if not (len(clauses) == len(references) == len(entities) == n):
            raise ValueError("clauses, outputs, references and entities must have the same length")
        parts = [
            self.meaning(clauses, outputs, references),
            self.simplicity(clauses, outputs, references),
            self.entity_preservation(outputs, entities),
        ]
        formats = [
            format_score(c, o, e, self.config.format)
            for c, o, e in zip(clauses, outputs, entities, strict=True)
        ]
        results = []
        for i in range(n):
            values = [parts[0][i], parts[1][i], parts[2][i], formats[i][0]]
            total = sum(w * v for w, v in zip(self.weights, values, strict=True))
            results.append(RewardBreakdown(*values, total=total, problems=tuple(formats[i][1])))
        return results

    def grpo_reward_functions(self) -> list[Callable[..., list[float]]]:
        """Callables in ``REWARD_NAMES`` order; TRL names each after the wrapped function."""

        return [
            functools.partial(fn, suite=self)
            for fn in (meaning, simplicity, entity_preservation, format)
        ]


# GRPO reward functions. TRL calls them with ``prompts``, ``completions`` and every
# dataset column as keyword arguments (here ``clause``, ``reference``, ``entities``),
# and logs each under its function name.


def meaning(
    prompts: Any,
    completions: Sequence[Any],
    *,
    suite: RewardSuite,
    clause: Sequence[str],
    reference: Sequence[str],
    **_: Any,
) -> list[float]:
    return suite.meaning(clause, [completion_text(c) for c in completions], reference)


def simplicity(
    prompts: Any,
    completions: Sequence[Any],
    *,
    suite: RewardSuite,
    clause: Sequence[str],
    reference: Sequence[str],
    **_: Any,
) -> list[float]:
    return suite.simplicity(clause, [completion_text(c) for c in completions], reference)


def entity_preservation(
    prompts: Any,
    completions: Sequence[Any],
    *,
    suite: RewardSuite,
    entities: Sequence[Sequence[str]],
    **_: Any,
) -> list[float]:
    return suite.entity_preservation([completion_text(c) for c in completions], entities)


def format(
    prompts: Any,
    completions: Sequence[Any],
    *,
    suite: RewardSuite,
    clause: Sequence[str],
    entities: Sequence[Sequence[str]],
    **_: Any,
) -> list[float]:
    return suite.format(clause, [completion_text(c) for c in completions], entities)
