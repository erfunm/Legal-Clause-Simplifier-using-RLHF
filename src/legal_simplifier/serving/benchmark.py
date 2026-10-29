"""``lcs benchmark``: latency, throughput and cost per 1,000 clauses for the served model.

Two passes over ``n_clauses`` held-out clauses, after ``warmup`` untimed calls:

* **latency**: one clause per call, as the API serves them: p50 / p95 / mean ms;
* **throughput**: ``batch_size`` clauses per call: completion tokens per second
  and seconds per clause.

Cost per 1,000 clauses = ``gpu_hourly_usd`` x (seconds per clause in the
throughput pass) x 1000 / 3600. It is the price of the GPU alone on the stated
GPU (the default config cites Modal's L4 price), so it is a lower bound: CPU,
memory, idle time and egress are not included.

The Claude comparison comes from the eval run's ``claude`` system: its real
spend, matched by request key in the cost log (Batch API prices). Without
that, an *estimate* from the Batch price table is reported and labelled as one:
input tokens counted locally from the exact requests, and output length
borrowed from the served model.

The result goes to ``results/benchmarks/<name>.json``; ``lcs report`` reads it
(non-smoke results only) for the quality-vs-cost chart.
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from legal_simplifier.config import BenchmarkConfig, EvalConfigFile, SystemSpec
from legal_simplifier.prompts import build_messages
from legal_simplifier.serving.engine import Engine, make_engine

LOGGER = logging.getLogger(__name__)


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def load_clauses(path: Path, n: int) -> list[tuple[str, str | None]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"no clauses in {path}")
    rows = (rows * (n // len(rows) + 1))[:n]  # repeat a small file to reach n
    return [(" ".join(r["clause"].split()), r.get("category")) for r in rows]


def claude_cost(
    config: BenchmarkConfig, clauses: list[tuple[str, str | None]], mean_output_tokens: float
) -> dict[str, Any]:
    """USD per 1,000 clauses for Claude: measured from the cost log if possible, else estimated."""

    from legal_simplifier.llm.batch import BATCH_DISCOUNT, PRICES_PER_MTOK

    if config.eval_run is not None and (config.eval_run / "manifest.json").exists():
        from legal_simplifier.eval.evaluate import load_run
        from legal_simplifier.eval.report import system_costs

        manifest, rows = load_run(config.eval_run)
        measured = system_costs(manifest, rows, config.cost_log).get("claude")
        if measured is not None:
            return {
                "usd_per_1k_clauses": measured,
                "source": "cost_log",
                "eval_run": str(config.eval_run),
            }

    from legal_simplifier.eval.items import EvalItem
    from legal_simplifier.eval.systems import ClaudeSystem
    from legal_simplifier.llm.estimate import LocalTokenCounter

    price = PRICES_PER_MTOK.get(config.claude_model)
    if price is None:
        return {"usd_per_1k_clauses": None, "source": f"no price for {config.claude_model}"}
    eval_config = EvalConfigFile.model_validate(
        {
            "experiment": {"name": "benchmark"},
            "systems": [],
            "claude": {"model": config.claude_model},
        }
    )
    system = ClaudeSystem(SystemSpec(name="claude", kind="claude"), eval_config)
    counter = LocalTokenCounter()
    inputs = [
        sum(counter.count(system.request(EvalItem("bench", str(i), c or "", text, "", "")))) / 2
        for i, (text, c) in enumerate(clauses)
    ]
    per_clause = (statistics.fmean(inputs) * price.input + mean_output_tokens * price.output) / 1e6
    return {
        "usd_per_1k_clauses": round(1000 * per_clause * BATCH_DISCOUNT, 4),
        "source": "batch_price_estimate",
        "note": "input tokens approximated locally; output length taken from the served model",
    }


def run_benchmark(config: BenchmarkConfig, *, engine: Engine | None = None) -> Path:
    engine = engine or make_engine(config.serve)
    clauses = load_clauses(config.clauses_path, config.n_clauses)
    max_new = config.serve.max_new_tokens
    messages = [build_messages(text, category) for text, category in clauses]

    for m in messages[: config.warmup]:
        engine.generate([m], max_new)

    latencies, completion = [], []
    for m in messages:
        start = time.perf_counter()
        (g,) = engine.generate([m], max_new)
        latencies.append((time.perf_counter() - start) * 1000)
        completion.append(g.completion_tokens)

    start = time.perf_counter()
    batch_tokens = 0
    for i in range(0, len(messages), config.batch_size):
        batch_tokens += sum(
            g.completion_tokens
            for g in engine.generate(messages[i : i + config.batch_size], max_new)
        )
    total_s = time.perf_counter() - start
    seconds_per_clause = total_s / len(messages)

    usd = None
    if config.gpu_hourly_usd is not None:
        usd = round(config.gpu_hourly_usd * seconds_per_clause * 1000 / 3600, 4)
    mean_out = statistics.fmean(completion)
    result = {
        "name": config.experiment.name,
        "smoke": config.experiment.smoke,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "system": config.serve.model_label,
        "model": config.serve.model.name_or_path,
        "engine": engine.name,
        "gpu": config.gpu,
        "n_clauses": len(messages),
        "max_new_tokens": max_new,
        "latency_ms": {
            "p50": round(_percentile(latencies, 0.5), 1),
            "p95": round(_percentile(latencies, 0.95), 1),
            "mean": round(statistics.fmean(latencies), 1),
        },
        "throughput": {
            "batch_size": config.batch_size,
            "completion_tokens_per_s": round(batch_tokens / total_s, 1) if total_s else None,
            "seconds_per_clause": round(seconds_per_clause, 4),
        },
        "mean_completion_tokens": round(mean_out, 1),
        "usd_per_1k_clauses": usd,
        "cost_basis": (
            f"GPU only at ${config.gpu_hourly_usd}/h ({config.gpu_price_source}); CPU, memory and idle time excluded"
            if usd is not None
            else "unknown: set gpu_hourly_usd"
        ),
        "claude": claude_cost(config, clauses, mean_out),
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    path = config.output_dir / f"{config.experiment.name}.json"
    path.write_text(json.dumps(result, indent=2))
    LOGGER.info("Benchmark written to %s", path)
    return path


def benchmark_costs(results_dir: Path | str) -> dict[str, float]:
    """System -> USD per 1,000 clauses from non-smoke benchmark files (newest wins)."""

    costs: dict[str, tuple[str, float]] = {}
    for path in sorted((Path(results_dir) / "benchmarks").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("smoke", True) or data.get("usd_per_1k_clauses") is None:
            continue
        current = costs.get(data["system"])
        if current is None or data["created_at"] > current[0]:
            costs[data["system"]] = (data["created_at"], data["usd_per_1k_clauses"])
    return {system: usd for system, (_, usd) in costs.items()}
