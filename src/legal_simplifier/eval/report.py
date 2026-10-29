"""``lcs report``: build the results tables and figures from ``results/``.

Every run under ``results/`` is checked by :func:`assert_publishable_run`
*before anything is written*, so mock references and smoke runs can never reach
the README or ``docs/figures/``. Then, from the primary run (``--run``, or the
most recent):

* the README section between ``<!-- results:start -->`` and
  ``<!-- results:end -->`` is replaced with the metrics table (mean and 95%
  bootstrap CI, best value in bold, ▲/▼ where a trained system differs from SFT
  with p < 0.05 in a paired test), the judge table, and mean ± std over seeds;
* ``docs/figures/metric_bars.png``: one panel per metric, CIs as error bars;
* ``docs/figures/quality_vs_cost.png``: quality against USD per 1,000 clauses.
  The Claude system's cost is summed from the cost log (matched by request key);
  local systems are marked "cost pending" until Phase 5 measures serving cost;
* ``docs/figures/grpo_reward_curves.png``: ``rewards/<name>/mean`` per step from
  ``log_history.json`` files or Trackio / W&B CSV exports.

The same tables are written to ``results/report.md``.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from legal_simplifier.eval.guards import UnpublishableResultsError, assert_publishable_run

START, END = "<!-- results:start -->", "<!-- results:end -->"
SKIP_DIRS = frozenset({"cache"})
SIGNIFICANCE = 0.05

# key, header, direction (+1 higher is better, -1 lower, 0 no "best"), format
COLUMNS: tuple[tuple[str, str, int, str], ...] = (
    ("sari", "SARI", 1, "{:.3f}"),
    ("bertscore", "BERTScore", 1, "{:.3f}"),
    ("fkgl", "FKGL (target 5-9)", 0, "{:.1f}"),  # a band, not "lower is better"
    ("entity_recall", "Entity recall", 1, "{:.3f}"),
    ("nli_faithfulness", "NLI faithful", 1, "{:.3f}"),
    ("nli_coverage", "NLI coverage", 1, "{:.3f}"),
    ("compression_ratio", "Compression", 0, "{:.2f}"),
    ("hallucinated", "Hallucination rate", -1, "{:.1%}"),
    ("detector/condition_drop", "Condition drop", -1, "{:.1%}"),
    ("detector/any", "Any detector", -1, "{:.1%}"),
)
JUDGE_COLUMNS = tuple(
    (f"judge/{c}", c.replace("_", " "), 1, "{:.2f}")
    for c in ("meaning_preserved", "no_added_facts", "plain_language", "obligations_intact")
)


def find_runs(results_dir: Path | str) -> list[Path]:
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        return []
    return sorted(p for p in results_dir.iterdir() if p.is_dir() and p.name not in SKIP_DIRS)


def _load(run: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    return manifest, metrics


# -- tables --------------------------------------------------------------------------


def _cell(stat: Mapping[str, Any] | None, fmt: str, *, bold: bool, mark: str) -> str:
    if not stat:
        return "\u2013"  # en dash
    text = fmt.format(stat["mean"])
    if stat["n"] > 1 and stat["low"] != stat["high"]:
        text += f" [{fmt.format(stat['low'])}, {fmt.format(stat['high'])}]"
    text = f"**{text}**" if bold else text
    return text + mark


def render_table(
    metrics: Mapping[str, Any],
    dataset: str,
    columns: Sequence[tuple[str, str, int, str]],
    systems: Sequence[str],
) -> str:
    present = [c for c in columns if any(metrics["systems"][s][dataset].get(c[0]) for s in systems)]
    if not present:
        return ""
    best: dict[str, float] = {}
    for key, _, direction, _ in present:
        values = [
            metrics["systems"][s][dataset][key]["mean"]
            for s in systems
            if metrics["systems"][s][dataset].get(key)
        ]
        if direction and values and len(set(values)) > 1:  # no bold when every system ties
            best[key] = max(values) if direction > 0 else min(values)
    lines = [
        "| system | " + " | ".join(h for _, h, _, _ in present) + " |",
        "|---|" + "---|" * len(present),
    ]
    for s in systems:
        cells = []
        for key, _, direction, fmt in present:
            stat = metrics["systems"][s][dataset].get(key)
            comparison = (metrics.get("comparisons", {}).get(s, {}).get(dataset) or {}).get(key)
            mark = ""
            if comparison and comparison["p_value"] < SIGNIFICANCE and direction:
                mark = " ▲" if comparison["diff"] * direction > 0 else " ▼"
            is_best = bool(stat) and key in best and stat["mean"] == best[key]
            cells.append(_cell(stat, fmt, bold=is_best, mark=mark))
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_seeds(metrics: Mapping[str, Any], dataset: str) -> str:
    if not metrics.get("seeds"):
        return ""
    keys = [
        c
        for c in (*COLUMNS, *JUDGE_COLUMNS)
        if c[0]
        in (
            "sari",
            "entity_recall",
            "nli_faithfulness",
            "detector/condition_drop",
            "judge/meaning_preserved",
        )
    ]
    lines = [
        "| group (seeds) | " + " | ".join(h for _, h, _, _ in keys) + " |",
        "|---|" + "---|" * len(keys),
    ]
    for group, data in metrics["seeds"].items():
        per = data.get(dataset, {})
        cells = []
        for key, _, _, fmt in keys:
            s = per.get(key)
            cells.append(f"{fmt.format(s['mean'])} ± {fmt.format(s['std'])}" if s else "\u2013")
        n = max((per.get(k[0]) or {}).get("n_seeds", 0) for k in keys)
        lines.append(f"| {group} ({n}) | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _dataset_order(datasets: Sequence[str]) -> list[str]:
    return sorted(datasets, key=lambda d: (d != "golden", d))


def render_section(
    manifest: Mapping[str, Any], metrics: Mapping[str, Any], figures: Mapping[str, str]
) -> str:
    systems = [s["name"] for s in manifest["systems"]]
    golden = manifest.get("golden_status") or {}
    parts = [
        f"*Generated by `lcs report` from `results/{manifest['run']}` "
        f"(created {manifest.get('created_at')}, commit `{(manifest.get('git_commit') or 'unknown')[:10]}`). "
        "Do not edit by hand.*",
        "",
        f"Golden set: {golden.get('n_items', '?')} clauses, review status **{golden.get('status', 'unknown')}**"
        + (
            " (references not yet all human-checked)."
            if golden.get("status") != "reviewed"
            else "."
        ),
        "Cells are means with 95% bootstrap CIs; the best value in each column is bold; "
        f"▲/▼: better/worse than `{metrics.get('reference_system')}` in a paired permutation test (p < {SIGNIFICANCE}).",
    ]
    for dataset in _dataset_order(metrics["datasets"]):
        parts += [
            "",
            f"#### {dataset} ({manifest['datasets'].get(dataset)} clauses)",
            "",
            render_table(metrics, dataset, COLUMNS, systems),
        ]
        judge = render_table(metrics, dataset, JUDGE_COLUMNS, systems)
        if judge and metrics.get("judge_backend") not in (None, "proxy"):
            parts += ["", f"LLM judge (`{metrics['judge_backend']}`, 1-5, blind):", "", judge]
        seeds = render_seeds(metrics, dataset)
        if seeds:
            parts += ["", "Mean ± std across training seeds:", "", seeds]
    for title, path in figures.items():
        parts += ["", f"![{title}]({path})"]
    return "\n".join(parts).strip() + "\n"


def replace_section(readme: str, section: str) -> str:
    if START not in readme or END not in readme:
        raise ValueError(f"README has no {START} ... {END} markers")
    head, rest = readme.split(START, 1)
    _, tail = rest.split(END, 1)
    return f"{head}{START}\n{section}{END}{tail}"


# -- figures --------------------------------------------------------------------------


def _plt() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_metric_bars(
    manifest: Mapping[str, Any], metrics: Mapping[str, Any], path: Path, dataset: str
) -> Path:
    plt = _plt()
    systems = [s["name"] for s in manifest["systems"]]
    cols = [
        c
        for c in (*COLUMNS, *JUDGE_COLUMNS)
        if any(metrics["systems"][s][dataset].get(c[0]) for s in systems)
    ]
    ncols = 4
    nrows = (len(cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    for ax, (key, header, _, _) in zip(axes.flat, cols, strict=False):
        stats = [metrics["systems"][s][dataset].get(key) for s in systems]
        means = [st["mean"] if st else 0 for st in stats]
        err = [
            [st["mean"] - st["low"] if st else 0 for st in stats],
            [st["high"] - st["mean"] if st else 0 for st in stats],
        ]
        ax.bar(range(len(systems)), means, yerr=err, capsize=3, color="#4C72B0")
        ax.set_title(header, fontsize=10)
        ax.set_xticks(range(len(systems)), systems, rotation=45, ha="right", fontsize=8)
    for ax in list(axes.flat)[len(cols) :]:
        ax.axis("off")
    fig.suptitle(f"{manifest['run']} · {dataset} (95% bootstrap CIs)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def system_costs(
    manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], cost_log: Path | None
) -> dict[str, float | None]:
    """USD per 1,000 clauses: Claude from the cost log (by request key); others pending."""

    from legal_simplifier.config import EvalConfigFile
    from legal_simplifier.eval.items import EvalItem
    from legal_simplifier.eval.systems import ClaudeSystem
    from legal_simplifier.llm.batch import CostLog, request_key

    costs: dict[str, float | None] = {s["name"]: None for s in manifest["systems"]}
    if cost_log is None or not Path(cost_log).exists():
        return costs
    by_key = {e["custom_id"]: e.get("cost_usd") or 0.0 for e in CostLog(cost_log).entries()}
    config = EvalConfigFile.model_validate(manifest["config"])
    for desc in manifest["systems"]:
        if desc["kind"] != "claude":
            continue
        spec = next(s for s in config.systems if s.name == desc["name"])
        claude = ClaudeSystem(spec, config)  # the run's own settings rebuild its exact requests
        mine = [r for r in rows if r["system"] == desc["name"]]
        keys = {
            request_key(
                claude.request(
                    EvalItem(
                        r["dataset"],
                        r["id"],
                        r["category"],
                        r["clause"],
                        r["reference"],
                        r["reference_source"],
                    )
                )
            )
            for r in mine
        }
        spent = [by_key[k] for k in keys if k in by_key]
        costs[desc["name"]] = 1000 * sum(spent) / len(spent) if spent else None
    return costs


def plot_quality_vs_cost(
    manifest: Mapping[str, Any],
    metrics: Mapping[str, Any],
    costs: Mapping[str, float | None],
    path: Path,
    dataset: str,
) -> Path:
    """Measured-cost systems on a scatter; systems without a cost yet in a side panel."""

    plt = _plt()
    judged = metrics.get("judge_backend") not in (None, "proxy")
    keys = [c[0] for c in JUDGE_COLUMNS] if judged else ["sari"]
    label = "judge score (mean of 4 criteria)" if judged else "SARI"
    measured, pending = [], []
    for desc in manifest["systems"]:
        name = desc["name"]
        per = metrics["systems"][name][dataset]
        values = [per[k]["mean"] for k in keys if per.get(k)]
        if values:
            quality = sum(values) / len(values)
            cost = costs.get(name)
            (pending if cost is None else measured).append((name, cost, quality))
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(10, 4.5), sharey=True, gridspec_kw={"width_ratios": [2, 3]}
    )
    if pending:
        left.scatter(range(len(pending)), [q for _, _, q in pending], marker="x", color="grey")
        left.set_xticks(
            range(len(pending)), [n for n, _, _ in pending], rotation=45, ha="right", fontsize=8
        )
    left.set_title("cost pending (Phase 5 serving numbers)", fontsize=10)
    left.set_ylabel(label)
    if measured:
        right.scatter([c for _, c, _ in measured], [q for _, _, q in measured], color="#C44E52")
        for name, cost, quality in measured:
            right.annotate(
                name, (cost, quality), fontsize=8, xytext=(4, 2), textcoords="offset points"
            )
        right.set_xlim(left=0)
    else:
        right.text(
            0.5, 0.5, "no measured costs yet", ha="center", va="center", transform=right.transAxes
        )
    right.set_xlabel("USD per 1,000 clauses (Batch API, from the cost log)")
    right.set_title("measured cost", fontsize=10)
    fig.suptitle(f"Quality vs cost · {dataset}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def read_training_log(path: Path) -> list[dict[str, float]]:
    """Steps with ``rewards/<name>/mean`` from ``log_history.json`` or a Trackio/W&B CSV export."""

    rows: list[dict[str, float]]
    if path.suffix == ".csv":
        with path.open(encoding="utf-8") as handle:
            rows = [
                {str(k): float(v) for k, v in r.items() if v not in ("", None)}
                for r in csv.DictReader(handle)
            ]
    else:
        rows = [
            {str(k): float(v) for k, v in r.items() if isinstance(v, int | float)}
            for r in json.loads(path.read_text())
        ]
    return [
        r
        for r in rows
        if "step" in r and any(k.startswith("rewards/") and k.endswith("/mean") for k in r)
    ]


def plot_reward_curves(logs: Sequence[Path], path: Path) -> Path | None:
    series = {p: read_training_log(p) for p in logs if p.exists()}
    series = {p: rows for p, rows in series.items() if rows}
    if not series:
        return None
    plt = _plt()
    names = sorted(
        {
            k
            for rows in series.values()
            for r in rows
            for k in r
            if k.startswith("rewards/") and k.endswith("/mean")
        }
    )
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.2), squeeze=False)
    for ax, name in zip(axes.flat, names, strict=True):
        for p, rows in series.items():
            points = [(r["step"], r[name]) for r in rows if name in r]
            if points:
                ax.plot(*zip(*points, strict=True), label=p.parent.name)
        ax.set_title(name.removeprefix("rewards/").removesuffix("/mean"), fontsize=10)
        ax.set_xlabel("step")
        ax.ticklabel_format(axis="y", useOffset=False)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("GRPO rewards (mean over each group)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# -- entry point ------------------------------------------------------------------------


@dataclass
class ReportResult:
    run: Path
    readme: Path | None
    report_md: Path
    figures: dict[str, Path] = field(default_factory=dict)


def generate_report(
    results_dir: Path | str,
    *,
    run: str | None = None,
    readme: Path | str | None = Path("README.md"),
    figures_dir: Path | str = Path("docs/figures"),
    training_logs: Sequence[Path | str] = (),
    cost_log: Path | str | None = Path("data/cache/anthropic/cost_log.jsonl"),
) -> ReportResult:
    results_dir = Path(results_dir)
    runs = find_runs(results_dir)
    if not runs:
        raise UnpublishableResultsError(f"no eval runs found under {results_dir}")
    for candidate in runs:  # every run, before anything is written
        assert_publishable_run(candidate)

    if run is not None:
        primary = results_dir / run
        if primary not in runs:
            raise ValueError(f"no run {run!r} under {results_dir}")
    else:
        primary = max(runs, key=lambda r: _load(r)[0].get("created_at", ""))
    manifest, metrics = _load(primary)
    from legal_simplifier.eval.evaluate import load_run

    _, rows = load_run(primary)

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    dataset = "golden" if "golden" in metrics["datasets"] else metrics["datasets"][0]
    figures: dict[str, Path] = {
        "Metrics per system": plot_metric_bars(
            manifest, metrics, figures_dir / "metric_bars.png", dataset
        ),
        "Quality vs cost": plot_quality_vs_cost(
            manifest,
            metrics,
            system_costs(manifest, rows, Path(cost_log) if cost_log else None),
            figures_dir / "quality_vs_cost.png",
            dataset,
        ),
    }
    curves = plot_reward_curves(
        [Path(p) for p in training_logs], figures_dir / "grpo_reward_curves.png"
    )
    if curves is not None:
        figures["GRPO reward curves"] = curves

    relative = {title: p.as_posix() for title, p in figures.items()}
    section = render_section(manifest, metrics, relative)
    report_md = results_dir / "report.md"
    report_md.write_text(f"# Results\n\n{section}", encoding="utf-8")
    readme_path = None
    if readme is not None:
        readme_path = Path(readme)
        readme_path.write_text(
            replace_section(readme_path.read_text(encoding="utf-8"), section), encoding="utf-8"
        )
    return ReportResult(primary, readme_path, report_md, figures)
