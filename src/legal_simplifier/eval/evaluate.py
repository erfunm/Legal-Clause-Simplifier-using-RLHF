"""``lcs evaluate``: run every system on the eval items, score them, write ``results/<run>/``.

Run directory (the format :mod:`legal_simplifier.eval.guards` checks):

* ``manifest.json``: run name, ``smoke`` flag, time, git commit, datasets and
  sizes, golden-set review status, every system's description and fingerprint,
  generation counts (cached / generated / failed) and the config.
* ``examples.jsonl``: one row per (dataset, item, system): the clause, reference,
  ``reference_source``, entities, the output and its metrics (with detector hits).
* ``metrics.json``: per system and dataset, the mean and bootstrap CI of every
  metric; paired tests of each trained system against ``reference_system``;
  mean ± std across seeds for grouped systems. Rebuilt by :func:`aggregate_run`
  whenever judgements are added.
* ``judgements_<backend>.jsonl``: written by ``lcs judge``.

Claude is the only paid system. :func:`paid_plan` lists what a run would submit
so the CLI can estimate the cost and refuse without ``--yes``.
"""

from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from legal_simplifier.config import EvalConfigFile
from legal_simplifier.eval.guards import EXAMPLES_FILE, MANIFEST_FILE
from legal_simplifier.eval.items import EvalItem, EvalSet, load_eval_items
from legal_simplifier.eval.metrics import (
    DETECTOR_METRICS,
    METRICS,
    Scorers,
    flat_metrics,
    score_outputs,
)
from legal_simplifier.eval.stats import bootstrap_ci, paired_test, seed_summary
from legal_simplifier.eval.systems import (
    ClaudeSystem,
    GenerationCache,
    System,
    generate_cached,
    make_system,
)

LOGGER = logging.getLogger(__name__)

METRICS_FILE = "metrics.json"
JUDGE_BACKENDS = ("claude", "hf", "proxy")  # primary judge = the first one present


def judgements_file(backend: str) -> str:
    return f"judgements_{backend}.jsonl"


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def build_systems(config: EvalConfigFile, *, client: Any | None = None) -> list[System]:
    return [make_system(spec, config, client=client) for spec in config.systems]


def paid_plan(
    systems: Sequence[System], items: Sequence[EvalItem], cache: GenerationCache
) -> list[tuple[ClaudeSystem, list[dict[str, Any]]]]:
    """For each paid system, the requests a run would submit (items not yet cached)."""

    plan = []
    for system in systems:
        if isinstance(system, ClaudeSystem):
            cached = cache.load(system)
            missing = [i for i in items if i.key not in cached]
            requests = (
                system.runner().unsubmitted([system.request(i) for i in missing]) if missing else []
            )
            plan.append((system, list(requests)))
    return plan


def run_evaluation(
    config: EvalConfigFile,
    *,
    run_name: str | None = None,
    limit: int | None = None,
    scorers: Scorers | None = None,
    systems: Sequence[System] | None = None,
    skip_paid: bool = False,
) -> Path:
    """Generate (or reuse cached) outputs for every system, score them and write the run."""

    eval_set = load_eval_items(config.data, limit=limit, seed=config.judge.pilot_seed)
    systems = list(systems) if systems is not None else build_systems(config)
    if skip_paid:
        systems = [s for s in systems if not s.paid]
    scorers = scorers or Scorers(config.metrics)
    cache = GenerationCache(config.cache_dir)
    run_dir = config.output_root / (run_name or config.experiment.name)
    run_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    for system in systems:
        outputs, counts[system.name] = generate_cached(system, eval_set.items, cache)
        scored = score_outputs(eval_set.items, outputs, scorers)
        for item, output, metrics in zip(eval_set.items, outputs, scored, strict=True):
            rows.append(
                {**item.to_dict(), "system": system.name, "output": output, "metrics": metrics}
            )

    manifest = {
        "run": run_dir.name,
        "smoke": config.experiment.smoke,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "datasets": eval_set.sizes,
        "limit": limit,
        "golden_status": eval_set.golden_status,
        "reference_system": config.reference_system,
        "systems": [s.describe() for s in systems],
        "generation": counts,
        "config": json.loads(config.model_dump_json()),
    }
    (run_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, default=str))
    with (run_dir / EXAMPLES_FILE).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    aggregate_run(run_dir, config)
    LOGGER.info("Wrote %s (%d rows, %d systems)", run_dir, len(rows), len(systems))
    return run_dir


# -- reading a run back ------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_run(run_dir: Path | str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
    return manifest, read_jsonl(run_dir / EXAMPLES_FILE)


def primary_judge(run_dir: Path | str) -> str | None:
    for backend in JUDGE_BACKENDS:
        if (Path(run_dir) / judgements_file(backend)).exists():
            return backend
    return None


def attach_judgements(
    rows: Sequence[dict[str, Any]], judgements: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Copy each row and add ``judge`` (criterion -> score) where a judgement exists."""

    scores = {(j["dataset"], j["id"], j["system"]): j["scores"] for j in judgements}
    return [{**r, "judge": scores.get((r["dataset"], r["id"], r["system"]))} for r in rows]


def scored_rows(run_dir: Path | str) -> tuple[dict[str, Any], list[dict[str, Any]], str | None]:
    """Manifest, rows with the primary judge's scores attached, and which judge that is."""

    manifest, rows = load_run(run_dir)
    backend = primary_judge(run_dir)
    if backend is not None:
        rows = attach_judgements(rows, read_jsonl(Path(run_dir) / judgements_file(backend)))
    return manifest, rows, backend


# -- aggregation ----------------------------------------------------------------------


def aggregate(
    rows: Sequence[dict[str, Any]],
    systems: Sequence[dict[str, Any]],
    *,
    reference_system: str,
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Means with bootstrap CIs, paired tests against ``reference_system``, seed summaries."""

    datasets = sorted({r["dataset"] for r in rows})
    table: dict[tuple[str, str], dict[str, dict[str, float | None]]] = {}
    for row in rows:
        table.setdefault((row["system"], row["dataset"]), {})[row["id"]] = flat_metrics(row)
    judged = {
        k for per in table.values() for m in per.values() for k in m if k.startswith("judge/")
    }
    metric_names = [*METRICS, *DETECTOR_METRICS, *sorted(judged)]
    stats = {"n_boot": n_boot, "confidence": confidence, "seed": seed}

    def interval(values: list[float | None]) -> dict[str, Any] | None:
        ci = bootstrap_ci(values, **stats)
        return ci.to_dict() if ci else None

    result: dict[str, Any] = {
        "datasets": datasets,
        "metrics": metric_names,
        "reference_system": reference_system,
        "systems": {},
        "comparisons": {},
        "seeds": {},
    }
    for system in systems:
        name = system["name"]
        result["systems"][name] = {}
        for dataset in datasets:
            per_item = table.get((name, dataset), {}).values()
            result["systems"][name][dataset] = {
                metric: interval([m.get(metric) for m in per_item]) for metric in metric_names
            }

    for system in systems:
        name = system["name"]
        if not system.get("trained") or name == reference_system:
            continue
        result["comparisons"][name] = {}
        for dataset in datasets:
            ours = table.get((name, dataset), {})
            ref = table.get((reference_system, dataset), {})
            shared = sorted(set(ours) & set(ref))
            per_metric: dict[str, Any] = {}
            for metric in metric_names:
                test = paired_test(
                    [ours[i].get(metric) for i in shared],
                    [ref[i].get(metric) for i in shared],
                    **stats,
                )
                per_metric[metric] = test.to_dict() if test else None
            result["comparisons"][name][dataset] = per_metric

    groups: dict[str, list[str]] = {}
    for system in systems:
        if system.get("group"):
            groups.setdefault(system["group"], []).append(system["name"])
    for group, members in groups.items():
        result["seeds"][group] = {"systems": members}
        for dataset in datasets:
            per_metric = {}
            for metric in metric_names:
                means = [(result["systems"][m][dataset][metric] or {}).get("mean") for m in members]
                per_metric[metric] = seed_summary(means)
            result["seeds"][group][dataset] = per_metric
    return result


def aggregate_run(run_dir: Path | str, config: EvalConfigFile | None = None) -> dict[str, Any]:
    """(Re)compute ``metrics.json`` from the run's examples and its primary judge."""

    run_dir = Path(run_dir)
    manifest, rows, backend = scored_rows(run_dir)
    m = config.metrics if config is not None else None
    metrics = aggregate(
        rows,
        manifest["systems"],
        reference_system=manifest.get("reference_system", "sft"),
        n_boot=m.n_bootstrap if m else 1000,
        confidence=m.confidence if m else 0.95,
        seed=config.experiment.seed if config else 0,
    )
    metrics["run"] = manifest["run"]
    metrics["smoke"] = manifest.get("smoke", True)
    metrics["judge_backend"] = backend
    (run_dir / METRICS_FILE).write_text(json.dumps(metrics, indent=2))
    return metrics


def eval_set_for_run(config: EvalConfigFile, limit: int | None = None) -> EvalSet:
    return load_eval_items(config.data, limit=limit, seed=config.judge.pilot_seed)
