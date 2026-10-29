"""``lcs judge-agreement``: how well the judge matches human labels on the same rubric.

For each criterion: Spearman's rank correlation and quadratic-weighted Cohen's
kappa between the human score and the judge score of the same output, each with
a percentile bootstrap CI. The bootstrap resamples *outputs* (all reviewers'
labels of an output move together), so several reviewers do not inflate n.
The same is reported between the primary (Claude) judge and the open-weight
cross-check when both have judged the run, which is where self-preference
would show up as a gap.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from legal_simplifier.eval.evaluate import judgements_file, read_jsonl
from legal_simplifier.eval.judge import CRITERIA
from legal_simplifier.eval.labels import latest_labels

AGREEMENT_FILE = "agreement.json"
MIN_RELIABLE_N = 30
# The acceptance bar (docs/decisions.md, ADR-026), fixed before any real labels: for each
# criterion, quadratic-weighted kappa and Spearman both at least MIN_VALUE, and the lower
# bound of each one's 95% bootstrap CI at least MIN_LOWER_BOUND.
MIN_VALUE = 0.6
MIN_LOWER_BOUND = 0.4
ACCEPTANCE = {
    "min_value": MIN_VALUE,
    "min_lower_bound": MIN_LOWER_BOUND,
    "statistics": ["quadratic_kappa", "spearman"],
}


def acceptance(row: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """Whether one criterion's agreement meets the bar, and the reasons it does not."""

    reasons = []
    for stat in ACCEPTANCE["statistics"]:
        value, low = row[stat]["value"], row[stat]["low"]
        if value is None or value < MIN_VALUE:
            reasons.append(f"{stat} {value if value is None else round(value, 2)} < {MIN_VALUE}")
        if low is None or low < MIN_LOWER_BOUND:
            reasons.append(
                f"{stat} CI lower bound {low if low is None else round(low, 2)} < {MIN_LOWER_BOUND}"
            )
    return not reasons, reasons


def _ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank), 1-based."""

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_vals = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    x, y = _ranks(np.asarray(a, dtype=float)), _ranks(np.asarray(b, dtype=float))
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def quadratic_kappa(a: Sequence[int], b: Sequence[int], *, k: int = 5) -> float | None:
    """Cohen's kappa with quadratic weights for ordinal scores 1..k."""

    x, y = np.asarray(a, dtype=int) - 1, np.asarray(b, dtype=int) - 1
    if len(x) == 0:
        return None
    observed = np.zeros((k, k))
    for i, j in zip(x, y, strict=True):
        observed[i, j] += 1
    weights = (
        np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)], dtype=float) / (k - 1) ** 2
    )
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
    denom = (weights * expected).sum()
    if denom == 0:
        return None
    return float(1 - (weights * observed).sum() / denom)


def _bootstrap(
    pairs: Sequence[tuple[str, int, int]], stat: Any, *, n_boot: int, seed: int, confidence: float
) -> dict[str, Any]:
    """Point estimate and a CI, resampling clusters (outputs), not individual labels."""

    point = stat([p[1] for p in pairs], [p[2] for p in pairs])
    clusters: dict[str, list[tuple[int, int]]] = {}
    for cid, h, j in pairs:
        clusters.setdefault(cid, []).append((h, j))
    keys = list(clusters)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        sample = [pair for key in rng.choice(keys, size=len(keys)) for pair in clusters[key]]
        value = stat([h for h, _ in sample], [j for _, j in sample])
        if value is not None:
            values.append(value)
    alpha = (1 - confidence) / 2
    low, high = (
        (float(np.quantile(values, alpha)), float(np.quantile(values, 1 - alpha)))
        if values
        else (None, None)
    )
    return {"value": point, "low": low, "high": high}


def agreement_table(
    pairs_by_criterion: Mapping[str, Sequence[tuple[str, int, int]]],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for criterion, pairs in pairs_by_criterion.items():
        n_outputs = len({p[0] for p in pairs})
        table[criterion] = {
            "n_labels": len(pairs),
            "n_outputs": n_outputs,
            "spearman": _bootstrap(
                pairs, spearman, n_boot=n_boot, seed=seed, confidence=confidence
            ),
            "quadratic_kappa": _bootstrap(
                pairs, quadratic_kappa, n_boot=n_boot, seed=seed, confidence=confidence
            ),
            "reliable": n_outputs >= MIN_RELIABLE_N,
        }
    return table


def _judge_scores(run_dir: Path, backend: str) -> dict[tuple[str, str, str], dict[str, int]]:
    return {
        (j["dataset"], j["id"], j["system"]): j["scores"]
        for j in read_jsonl(run_dir / judgements_file(backend))
        if j.get("backend") != "rule"  # empty outputs scored by rule are not the judge's call
    }


def judge_agreement(
    run_dir: Path | str,
    *,
    backend: str = "claude",
    other: str | None = "hf",
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    judge = _judge_scores(run_dir, backend)
    if not judge:
        raise FileNotFoundError(f"no {backend} judgements in {run_dir}: run `lcs judge` first")
    labels = latest_labels(run_dir)
    if not labels:
        raise FileNotFoundError(f"no human labels in {run_dir}: run `lcs label-judge` first")
    human_pairs: dict[str, list[tuple[str, int, int]]] = {c: [] for c in CRITERIA}
    for lab in labels:
        key = (lab["dataset"], lab["id"], lab["system"])
        if key in judge:
            for c in CRITERIA:
                human_pairs[c].append((lab["queue_id"], int(lab["scores"][c]), int(judge[key][c])))
    table = agreement_table(human_pairs, n_boot=n_boot, seed=seed)
    for row in table.values():
        row["passes"], row["fails_because"] = acceptance(row)
    result: dict[str, Any] = {
        "judge": backend,
        "reviewers": sorted({lab["reviewer"] for lab in labels}),
        "acceptance": ACCEPTANCE,
        "human_vs_judge": table,
        "validated_criteria": sorted(c for c, row in table.items() if row["passes"]),
    }
    second = _judge_scores(run_dir, other) if other else {}
    if second:
        judge_pairs: dict[str, list[tuple[str, int, int]]] = {c: [] for c in CRITERIA}
        for key, scores in judge.items():
            if key in second:
                for c in CRITERIA:
                    judge_pairs[c].append(("\x1f".join(key), int(scores[c]), int(second[key][c])))
        result[f"{backend}_vs_{other}"] = agreement_table(judge_pairs, n_boot=n_boot, seed=seed)
    (run_dir / AGREEMENT_FILE).write_text(json.dumps(result, indent=2))
    return result


def validated_criteria(run_dir: Path | str, backend: str | None) -> set[str]:
    """Judge criteria that passed the bar for ``backend`` in this run (empty if never checked)."""

    path = Path(run_dir) / AGREEMENT_FILE
    if backend is None or not path.exists():
        return set()
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("judge") != backend:
        return set()
    return set(result.get("validated_criteria", []))


def describe(result: Mapping[str, Any]) -> str:
    bar = result.get("acceptance")
    lines = (
        [
            f"Acceptance bar per criterion: QWK and Spearman >= {bar['min_value']}, "
            f"and each 95% CI lower bound >= {bar['min_lower_bound']}."
        ]
        if bar
        else []
    )
    for section, table in result.items():
        if not isinstance(table, dict) or section == "acceptance":
            continue
        lines.append(f"{section}:")
        for criterion, row in table.items():

            def fmt(stat: Mapping[str, Any]) -> str:
                if stat["value"] is None:
                    return "n/a"
                ci = f" [{stat['low']:.2f}, {stat['high']:.2f}]" if stat["low"] is not None else ""
                return f"{stat['value']:.2f}{ci}"

            flag = "" if row["reliable"] else f"  (only {row['n_outputs']} outputs: not reliable)"
            verdict = ""
            if "passes" in row:
                verdict = (
                    "  PASS" if row["passes"] else "  FAIL: " + "; ".join(row["fails_because"])
                )
            lines.append(
                f"  {criterion}: Spearman {fmt(row['spearman'])}, QWK {fmt(row['quadratic_kappa'])}, "
                f"n={row['n_labels']}{flag}{verdict}"
            )
    return "\n".join(lines)
