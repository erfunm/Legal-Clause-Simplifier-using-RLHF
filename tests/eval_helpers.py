"""Shared builders for the evaluation tests: a golden set, an eval config, fake scorers."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from legal_simplifier.config import EvalConfigFile
from legal_simplifier.data.golden import build_golden
from legal_simplifier.eval.metrics import Scorers
from legal_simplifier.rewards.sari import tokenize

from .conftest import make_record

FIXTURES = Path(__file__).parent / "fixtures"

TERMINATION = (
    "The Licensee may terminate on 30 days' notice, unless the Licensor has cured the breach "
    "within 10 days."
)
FAITHFUL = (
    "The Licensee can end this agreement with 30 days' notice, but not if the Licensor fixes "
    "the breach within 10 days."
)
DROPS = "The Licensee can end this agreement with 30 days' notice."

CLAUSES = [
    ("Termination For Convenience", TERMINATION, FAITHFUL),
    (
        "Cap On Liability",
        "In no event shall Acme Inc. be liable for more than $250,000, except for gross negligence.",
        "Acme Inc. must not pay more than $250,000, except for gross negligence.",
    ),
    (
        "Governing Law",
        "This Agreement shall be governed by the laws of the State of New York.",
        "The laws of the State of New York apply to this Agreement.",
    ),
    (
        "Non-Compete",
        "During the Term, the Distributor shall not sell any product that competes with the Products.",
        "During the Term, the Distributor must not sell products that compete with the Products.",
    ),
]


class FakeNLI:
    """Entailment = share of hypothesis content words found in the premise."""

    def __init__(self) -> None:
        self.calls = 0

    def entailment(self, premises: Sequence[str], hypotheses: Sequence[str]) -> list[float]:
        self.calls += 1
        out = []
        for p, h in zip(premises, hypotheses, strict=True):
            words = set(tokenize(h))
            out.append(len(words & set(tokenize(p))) / len(words) if words else 0.0)
        return out


class FakeSimilarity:
    def score(self, predictions: Sequence[str], references: Sequence[str]) -> list[float]:
        out = []
        for p, r in zip(predictions, references, strict=True):
            a, b = set(tokenize(p)), set(tokenize(r))
            out.append(2 * len(a & b) / (len(a) + len(b)) if a or b else 0.0)
        return out


def make_golden(root: Path, *, source: str = "claude") -> Path:
    records = [
        make_record(
            f"t{i}", f"k{i}", clause, category=category, reference=ref, reference_source=source
        ).model_copy(update={"split": "test"})
        for i, (category, clause, ref) in enumerate(CLAUSES)
    ]
    from legal_simplifier.data.entities import extract_entities

    records = [r.model_copy(update={"entities": extract_entities(r.clause)}) for r in records]
    golden_dir = root / "golden"
    build_golden(records, golden_dir, size=len(records))
    return golden_dir


def eval_config(
    root: Path, *, systems: list[dict[str, Any]] | None = None, smoke: bool = False, **extra: Any
) -> EvalConfigFile:
    golden_dir = root / "golden" if (root / "golden").exists() else make_golden(root)
    return EvalConfigFile.model_validate(
        {
            "experiment": {"name": "test-eval", "seed": 0, "smoke": smoke},
            "output_root": root / "results",
            "cache_dir": root / "results" / "cache" / "generations",
            "data": {"golden_dir": golden_dir},
            "systems": systems
            or [
                {"name": "copy", "kind": "copy"},
                {"name": "truncate", "kind": "truncate"},
            ],
            "metrics": {"n_bootstrap": 200},
            "claude": {
                "cache_dir": root / "cache",
                "cost_log": root / "cache" / "cost.jsonl",
                "poll_interval_s": 0,
            },
            "judge": {
                "cache_dir": root / "cache",
                "cost_log": root / "cache" / "cost.jsonl",
                "poll_interval_s": 0,
            },
            **extra,
        }
    )


def fake_scorers(config: EvalConfigFile) -> Scorers:
    return Scorers(config.metrics, entailment=FakeNLI(), similarity=FakeSimilarity())


def judge_outputs() -> dict[str, Any]:
    return json.loads((FIXTURES / "anthropic" / "judge_outputs.json").read_text())


def judgement_json(labels_to_kind: dict[str, str]) -> str:
    """A judge response attaching the hand-written gradings to the given labels."""

    templates = judge_outputs()
    return json.dumps(
        {"outputs": [{"label": label, **templates[kind]} for label, kind in labels_to_kind.items()]}
    )
