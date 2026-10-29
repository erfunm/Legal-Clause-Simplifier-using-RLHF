"""``lcs eval-gate``: re-score a frozen subset of stored outputs and fail on regressions.

The fixture (``tests/fixtures/eval_gate/``) is 20 clauses with **mock** data:
synthetic clauses and rule-made outputs, labelled as such. It holds outputs, not
a model, so the gate needs no GPU, no download and no API. What it guards is
the scoring code: it re-runs the entity checks, the reward-hacking detectors and
the judge *proxy* (deterministic 1-5 stand-ins, :func:`eval.judge.proxy_scores`)
and fails when a metric moves outside ``thresholds.json``. A change that stops
the detectors firing, or a worse set of outputs, trips it; the tests show both.

Thresholds are ``{"metric": {"min": x}}`` or ``{"max": y}``. Metrics:
``entity_recall`` (mean), ``detector/<name>`` (rate) and
``judge_proxy/<criterion>`` (mean).
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from legal_simplifier.eval.judge import CRITERIA, proxy_scores
from legal_simplifier.rewards.components import entity_preservation_score
from legal_simplifier.rewards.hacking import DETECTORS, run_detectors

FIXTURE_DIR = Path("tests/fixtures/eval_gate")


@dataclass
class GateResult:
    metrics: dict[str, float]
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures


def gate_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("the gate fixture is empty")
    recalls, proxies = [], {c: [] for c in CRITERIA}
    hits = {d: 0 for d in DETECTORS}
    for row in rows:
        entities = row.get("entities") or []
        output = row["output"]
        recalls.append(entity_preservation_score(entities, output))
        for name, finding in run_detectors(row["clause"], output, entities).items():
            hits[name] += finding.flagged
        for c, v in proxy_scores(row["clause"], output, entities).items():
            proxies[c].append(v)
    metrics = {"entity_recall": statistics.fmean(recalls)}
    metrics |= {f"detector/{d}": n / len(rows) for d, n in hits.items()}
    metrics |= {f"judge_proxy/{c}": statistics.fmean(v) for c, v in proxies.items()}
    return metrics


def check(metrics: Mapping[str, float], thresholds: Mapping[str, Mapping[str, float]]) -> list[str]:
    failures = []
    for name, bounds in thresholds.items():
        if name not in metrics:
            failures.append(f"{name}: not computed (unknown metric in thresholds)")
            continue
        value = metrics[name]
        if "min" in bounds and value < bounds["min"]:
            failures.append(f"{name} = {value:.3f} is below the minimum {bounds['min']}")
        if "max" in bounds and value > bounds["max"]:
            failures.append(f"{name} = {value:.3f} is above the maximum {bounds['max']}")
    return failures


def run_gate(examples: Path | str, thresholds: Path | str) -> GateResult:
    with Path(examples).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    limits = json.loads(Path(thresholds).read_text(encoding="utf-8"))["thresholds"]
    metrics = gate_metrics(rows)
    return GateResult(metrics, check(metrics, limits))
