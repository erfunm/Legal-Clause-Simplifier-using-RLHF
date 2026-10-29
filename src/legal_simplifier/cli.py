"""``lcs`` command-line interface."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, TypeVar

import typer

T = TypeVar("T")

app = typer.Typer(
    help="Legal Clause Simplifier: data prep, post-training, evaluation and serving.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def _main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _not_yet(stage: str, phase: int) -> None:
    typer.echo(f"`lcs {stage}` is not implemented yet (planned for Phase {phase}).", err=True)
    raise typer.Exit(code=2)


DataConfigOption = Annotated[
    Path, typer.Option("--config", help="Data pipeline YAML (configs/data.yaml).")
]


def _data_config(path: Path):  # returns DataPrepConfig (imported lazily)
    from legal_simplifier.config import DataPrepConfig, load_config

    return load_config(path, DataPrepConfig)


@app.command("prepare-data")
def prepare_data(
    config: DataConfigOption = Path("configs/data.yaml"),
    input_path: Annotated[
        Path | None,
        typer.Option(
            "--input", help="Split existing records (JSONL with references) instead of CUAD."
        ),
    ] = None,
    output_dir: Annotated[Path | None, typer.Option(help="Override output_dir.")] = None,
) -> None:
    """Extract CUAD clauses, filter, deduplicate and split them by contract."""

    from legal_simplifier.data.prepare import prepare_from_cuad, prepare_from_jsonl

    cfg = _data_config(config)
    out = output_dir or cfg.output_dir
    split_kwargs = {
        "dedup_threshold": cfg.split.dedup_threshold,
        "ratios": cfg.split.ratios,
        "seed": cfg.split.seed,
    }
    if input_path is not None:
        result = prepare_from_jsonl(input_path, out, **split_kwargs)
        typer.echo(f"Wrote splits and stats to {result.output_dir}")
        return
    ex = cfg.extraction
    extra = {"categories": ex.categories} if ex.categories else {}
    result = prepare_from_cuad(
        out,
        cuad_json=ex.cuad_json,
        min_chars=ex.min_chars,
        min_words=ex.min_words,
        max_chars=ex.max_chars,
        **extra,
        **split_kwargs,
    )
    typer.echo(f"Wrote {len(result.records)} clause candidates and stats to {result.output_dir}")


@app.command("generate-references")
def generate_references(
    config: DataConfigOption = Path("configs/data.yaml"),
    backend: Annotated[str | None, typer.Option(help="mock | claude (overrides config).")] = None,
    dry_run: Annotated[
        bool,
        typer.Option(help="Send nothing; print the request count and an estimated cost range."),
    ] = False,
    yes: Annotated[
        bool, typer.Option("--yes", help="Confirm submitting paid API requests.")
    ] = False,
    limit: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Pilot: only a category-balanced sample of N clauses (seed: "
            "references.pilot_seed). Writes pilot_references.jsonl, not the split files.",
        ),
    ] = None,
    count_tokens: Annotated[
        str,
        typer.Option(
            help="How --dry-run counts input tokens: local (offline approximation) or api "
            "(Anthropic's free token-counting endpoint; needs ANTHROPIC_API_KEY).",
        ),
    ] = "local",
) -> None:
    """Attach silver references to the clause candidates and write train/dev/test files."""

    from legal_simplifier.data import references
    from legal_simplifier.llm.estimate import describe

    if count_tokens not in ("local", "api"):
        raise typer.BadParameter("must be local or api", param_hint="--count-tokens")
    cfg = _data_config(config)
    if not (cfg.output_dir / references.CANDIDATES_FILE).exists():
        typer.echo(
            f"error: {cfg.output_dir / references.CANDIDATES_FILE} not found; "
            "run `lcs prepare-data` first.",
            err=True,
        )
        raise typer.Exit(code=1)
    if backend is not None:
        cfg.references.backend = backend
    refused = cfg.references.backend == "claude" and not yes and not dry_run
    dry_run = dry_run or refused
    generator = references.make_generator(
        cfg.references,
        dry_run=dry_run,
        count_tokens="api" if count_tokens == "api" else "local",
    )
    _, report = references.generate_references(
        cfg.output_dir, generator, limit=limit, seed=cfg.references.pilot_seed
    )
    if report.batch is not None:
        typer.echo(references.summarise_run(report.batch))
    if report.estimate is not None:
        typer.echo(describe(report.estimate))
    if dry_run:
        typer.echo(
            "Nothing was sent. Re-run with --yes to submit the requests above "
            "(this calls the paid Anthropic API).",
            err=True,
        )
        raise typer.Exit(code=1 if refused else 0)
    typer.echo(
        f"{report.n_with_reference} of {report.n_candidates} clauses have references "
        f"({report.backend}); failures: {report.failures or 'none'}"
    )
    if limit is not None:
        typer.echo(f"Pilot written to {cfg.output_dir / references.PILOT_FILE}")


@app.command("build-golden")
def build_golden(config: DataConfigOption = Path("configs/data.yaml")) -> None:
    """Select the golden test set (marked unreviewed) from the test split."""

    from legal_simplifier.data.golden import build_golden as build
    from legal_simplifier.data.schema import read_records

    cfg = _data_config(config)
    test = read_records(cfg.output_dir / "test.jsonl")
    items = build(test, cfg.golden.output_dir, size=cfg.golden.size, seed=cfg.golden.seed)
    typer.echo(f"Wrote {len(items)} unreviewed golden items to {cfg.golden.output_dir}")


@app.command("review")
def review(
    reviewer: Annotated[str, typer.Option(help="Your name, recorded with every decision.")],
    config: DataConfigOption = Path("configs/data.yaml"),
    port: Annotated[int, typer.Option()] = 7860,
) -> None:
    """Open the golden-set review page (local only)."""

    from legal_simplifier.data.review_app import build_app

    cfg = _data_config(config)
    build_app(cfg.golden.output_dir, reviewer).launch(server_port=port, share=False)


@app.command("dataset-card")
def dataset_card(config: DataConfigOption = Path("configs/data.yaml")) -> None:
    """Write the dataset card (README.md) from stats.json and the golden-set status."""

    from legal_simplifier.data.card import write_dataset_card

    cfg = _data_config(config)
    typer.echo(f"Wrote {write_dataset_card(cfg.output_dir, cfg.golden.output_dir)}")


@app.command("push-dataset")
def push_dataset(
    repo_id: Annotated[
        str, typer.Option(help="Hub dataset repo, e.g. user/legal-clause-simplification.")
    ],
    config: DataConfigOption = Path("configs/data.yaml"),
    public: Annotated[
        bool, typer.Option(help="Create the repo as public (default private).")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", help="Confirm publishing to the Hub.")] = False,
) -> None:
    """Upload the processed dataset and card to the Hugging Face Hub. Refuses mock references."""

    from legal_simplifier.data.card import PublishRefused
    from legal_simplifier.data.card import push_dataset as push
    from legal_simplifier.eval.guards import UnpublishableResultsError

    cfg = _data_config(config)
    try:
        url = push(
            cfg.output_dir,
            repo_id,
            confirm=yes,
            private=not public,
            golden_dir=cfg.golden.output_dir,
        )
    except (PublishRefused, UnpublishableResultsError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Pushed to {url}")


@app.command("train-sft")
def train_sft(
    config: Annotated[Path, typer.Option("--config", help="SFT experiment YAML.")] = Path(
        "configs/sft.yaml"
    ),
    output_dir: Annotated[Path | None, typer.Option(help="Override training.output_dir.")] = None,
) -> None:
    """Supervised fine-tuning with TRL SFTTrainer: LoRA/QLoRA, completion-only loss."""

    from legal_simplifier.config import SFTConfigFile, load_config
    from legal_simplifier.training.sft import train_sft as run
    from legal_simplifier.training.tracking import TrackingPublishRefused

    cfg = load_config(config, SFTConfigFile)
    if output_dir is not None:
        cfg.training.output_dir = output_dir
    try:
        run_dir = run(cfg)
    except TrackingPublishRefused as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Saved SFT run to {run_dir}")


@app.command("merge-adapter")
def merge_adapter(
    run_dir: Annotated[Path, typer.Option(help="An SFT run directory (holds adapter/).")],
    output_dir: Annotated[Path | None, typer.Option(help="Default: <run-dir>/merged.")] = None,
) -> None:
    """Merge a LoRA adapter into its base model (loaded in its training dtype, never 4 bit)."""

    import json

    from legal_simplifier.config import ModelConfig
    from legal_simplifier.training.model_card import write_model_cards
    from legal_simplifier.training.sft import ADAPTER_DIR, MERGED_DIR
    from legal_simplifier.training.sft import merge_adapter as merge

    experiment = json.loads((run_dir / "experiment.json").read_text())  # any stage
    model = ModelConfig.model_validate(experiment["model"])
    out = merge(run_dir / ADAPTER_DIR, output_dir or run_dir / MERGED_DIR, model)
    write_model_cards(run_dir)
    typer.echo(f"Wrote merged weights to {out}")


@app.command("model-card")
def model_card(
    run_dir: Annotated[Path, typer.Option(help="An SFT run directory.")],
) -> None:
    """(Re)write README.md model cards for a run from its experiment, metrics and data stats."""

    from legal_simplifier.training.model_card import write_model_cards

    for path in write_model_cards(run_dir):
        typer.echo(f"Wrote {path}")


def _run_training(run: Callable[[], T]) -> T:
    from legal_simplifier.training.tracking import TrackingPublishRefused

    try:
        return run()
    except TrackingPublishRefused as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("train-grpo")
def train_grpo(
    config: Annotated[Path, typer.Option("--config", help="GRPO YAML.")] = Path(
        "configs/grpo.yaml"
    ),
    output_dir: Annotated[Path | None, typer.Option(help="Override grpo.output_dir.")] = None,
) -> None:
    """GRPO from the SFT adapter with four separately logged reward functions."""

    from legal_simplifier.config import GRPOConfigFile, load_config
    from legal_simplifier.training.grpo import train_grpo as run

    cfg = load_config(config, GRPOConfigFile)
    if output_dir is not None:
        cfg.grpo.output_dir = output_dir
    typer.echo(f"Saved GRPO run to {_run_training(lambda: run(cfg))}")


@app.command("build-preferences")
def build_preferences(
    config: Annotated[Path, typer.Option("--config", help="DPO YAML.")] = Path("configs/dpo.yaml"),
) -> None:
    """Sample N outputs per clause from the SFT model and keep (best, worst) pairs by reward."""

    from legal_simplifier.config import DPOConfigFile, load_config
    from legal_simplifier.training.preferences import build_preferences as build

    cfg = load_config(config, DPOConfigFile)
    for split, stats in build(cfg).items():
        typer.echo(
            f"{split}: {stats.n_pairs} pairs from {stats.n_clauses} clauses "
            f"(small margin {stats.skipped_small_margin}, identical {stats.skipped_identical}, "
            f"no clean sample {stats.skipped_no_clean_sample})"
        )
    typer.echo(f"Wrote pairs to {cfg.pairs.output_dir}")


@app.command("train-dpo")
def train_dpo(
    config: Annotated[Path, typer.Option("--config", help="DPO YAML.")] = Path("configs/dpo.yaml"),
    output_dir: Annotated[Path | None, typer.Option(help="Override dpo.output_dir.")] = None,
) -> None:
    """DPO from the SFT adapter on the pairs from `lcs build-preferences`."""

    from legal_simplifier.config import DPOConfigFile, load_config
    from legal_simplifier.training.dpo import train_dpo as run

    cfg = load_config(config, DPOConfigFile)
    if output_dir is not None:
        cfg.dpo.output_dir = output_dir
    try:
        typer.echo(f"Saved DPO run to {_run_training(lambda: run(cfg))}")
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("ablate")
def ablate(
    directory: Annotated[Path, typer.Option("--dir", help="Folder of ablation YAMLs.")] = Path(
        "configs/ablations"
    ),
    only: Annotated[
        list[str] | None, typer.Option(help="Run only these names (repeatable).")
    ] = None,
    dry_run: Annotated[bool, typer.Option(help="Print the resolved runs; train nothing.")] = False,
) -> None:
    """Run the GRPO ablations one after another on this machine (Modal runs them in parallel)."""

    from legal_simplifier.training.ablations import describe, discover, run_ablations

    ablations = discover(directory, only or ())
    for ablation in ablations:
        typer.echo(describe(ablation))
        typer.echo(f"    -> {ablation.config.grpo.output_dir}")
    if dry_run:
        typer.echo(f"{len(ablations)} ablation runs resolved; nothing was trained.")
        return
    for path in _run_training(lambda: run_ablations(ablations)):
        typer.echo(f"Saved {path}")


@app.command("detect-hacking")
def detect_hacking(
    outputs: Annotated[Path, typer.Option(help="JSONL of saved outputs (clause + output(s)).")],
    out_dir: Annotated[Path | None, typer.Option(help="Default: <outputs>_hacking/.")] = None,
) -> None:
    """Run the reward-hacking detectors over saved outputs; write summary.json and flagged.jsonl."""

    from legal_simplifier.eval.hacking_report import scan_file

    target = out_dir or outputs.with_name(f"{outputs.stem}_hacking")
    summary = scan_file(outputs, target).to_dict()
    typer.echo(f"{summary['n_outputs']} outputs, {summary['n_any_flagged']} flagged:")
    for name, row in summary["detectors"].items():
        typer.echo(f"  {name}: {row['count']} ({row['rate']:.1%})")
    typer.echo(f"Wrote {target / 'summary.json'} and {target / 'flagged.jsonl'}")


EvalConfigOption = Annotated[Path, typer.Option("--config", help="Eval YAML (configs/eval.yaml).")]
CountTokensOption = Annotated[
    str,
    typer.Option(
        help="How --dry-run counts input tokens: local (offline approximation) or api "
        "(Anthropic's free token-counting endpoint; needs ANTHROPIC_API_KEY).",
    ),
]


def _counter(count_tokens: str, client: Any | None) -> Any:
    from legal_simplifier.llm.estimate import ApiTokenCounter, LocalTokenCounter

    if count_tokens not in ("local", "api"):
        raise typer.BadParameter("must be local or api", param_hint="--count-tokens")
    return (
        ApiTokenCounter(client.messages)
        if count_tokens == "api" and client is not None
        else LocalTokenCounter()
    )


def _gate_paid(
    label: str,
    runner: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    model: str,
    counter: Any,
    dry_run: bool,
    yes: bool,
) -> None:
    """Print what would be sent and its cost range; exit unless the user said --yes."""

    from legal_simplifier.llm.estimate import describe, estimate_cost

    measured = [
        int(e["usage"].get("output_tokens") or 0)
        for e in runner.cost_log.entries()
        if e.get("model") == model and e.get("usage")
    ]
    typer.echo(f"{label}: {len(requests)} requests would be submitted (cached ones are free).")
    estimate = estimate_cost(requests, counter, measured_output_tokens=measured)
    if estimate is not None:
        typer.echo(describe(estimate))
    if dry_run:
        typer.echo("Dry run: nothing was sent.", err=True)
        raise typer.Exit(code=0)
    if requests and not yes:
        typer.echo(
            "Refusing to call the paid Anthropic API without --yes (nothing was sent).", err=True
        )
        raise typer.Exit(code=1)


@app.command("evaluate")
def evaluate(
    config: EvalConfigOption = Path("configs/eval.yaml"),
    run: Annotated[str | None, typer.Option(help="Run name (default: experiment.name).")] = None,
    limit: Annotated[
        int | None, typer.Option(min=1, help="Pilot: N items per dataset, category-balanced.")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option(help="Send and generate nothing; print the plan and cost.")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", help="Confirm the paid Claude system.")] = False,
    skip_paid: Annotated[
        bool, typer.Option(help="Leave out paid systems (Claude) this run.")
    ] = False,
    count_tokens: CountTokensOption = "local",
) -> None:
    """Run every system on the golden set (and dev), score them, write results/<run>/."""

    from legal_simplifier.config import EvalConfigFile, load_config
    from legal_simplifier.eval.evaluate import (
        build_systems,
        eval_set_for_run,
        paid_plan,
        run_evaluation,
    )
    from legal_simplifier.eval.systems import GenerationCache

    cfg = load_config(config, EvalConfigFile)
    systems = build_systems(cfg)
    if skip_paid:
        systems = [s for s in systems if not s.paid]
    items = eval_set_for_run(cfg, limit).items
    cache = GenerationCache(cfg.cache_dir)
    for system in systems:
        cached = cache.load(system)
        n_cached = sum(i.key in cached for i in items)
        typer.echo(f"{system.name} [{system.spec.kind}]: {n_cached} of {len(items)} outputs cached")
    for system, requests in paid_plan(systems, items, cache):
        _gate_paid(
            f"Claude system '{system.name}' ({cfg.claude.model})",
            system.runner(),
            requests,
            model=cfg.claude.model,
            counter=_counter(count_tokens, system.client),
            dry_run=dry_run,
            yes=yes,
        )
    if dry_run:
        typer.echo("Dry run: nothing was generated.")
        return
    run_dir = run_evaluation(cfg, run_name=run, limit=limit, systems=systems)
    typer.echo(f"Wrote {run_dir}")


@app.command("judge")
def judge(
    run_dir: Annotated[Path, typer.Option("--run", help="An eval run directory (results/<run>).")],
    config: EvalConfigOption = Path("configs/eval.yaml"),
    backend: Annotated[
        str | None, typer.Option(help="claude | hf | proxy (overrides judge.backend).")
    ] = None,
    limit: Annotated[
        int | None, typer.Option(min=1, help="Pilot: judge N clauses, category-balanced.")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option(help="Send nothing; print the request count and cost.")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", help="Confirm paid Claude judge calls.")] = False,
    count_tokens: CountTokensOption = "local",
) -> None:
    """Blind LLM-as-judge on the rubric; writes judgements_<backend>.jsonl and updates metrics.json."""

    from legal_simplifier.config import EvalConfigFile, load_config
    from legal_simplifier.eval.evaluate import aggregate_run, load_run
    from legal_simplifier.eval.judge import ClaudeJudge, make_judge, select_tasks, write_judgements

    cfg = load_config(config, EvalConfigFile)
    if backend is not None:
        cfg.judge.backend = backend  # type: ignore[assignment]
        cfg = EvalConfigFile.model_validate(cfg.model_dump())
    _, rows = load_run(run_dir)
    tasks = select_tasks(rows, seed=cfg.judge.seed, limit=limit, pilot_seed=cfg.judge.pilot_seed)
    judge_ = make_judge(cfg.judge)
    if isinstance(judge_, ClaudeJudge):
        _gate_paid(
            f"Judge ({cfg.judge.model}) on {len(tasks)} clauses",
            judge_.runner,
            list(judge_.runner.unsubmitted(judge_.requests(tasks))),
            model=cfg.judge.model,
            counter=_counter(count_tokens, judge_.client),
            dry_run=dry_run,
            yes=yes,
        )
    elif dry_run:
        typer.echo(
            f"Dry run: {len(tasks)} clauses would be judged by the {judge_.backend} backend."
        )
        return
    result = judge_.judge(tasks)
    path = write_judgements(run_dir, judge_.backend, result.rows)
    aggregate_run(run_dir, cfg)
    typer.echo(
        f"Wrote {len(result.rows)} judgements to {path}; failures: {result.failures or 'none'}"
    )


@app.command("label-judge")
def label_judge(
    run_dir: Annotated[Path, typer.Option("--run", help="An eval run directory.")],
    reviewer: Annotated[str, typer.Option(help="Your name, recorded with every label.")],
    n: Annotated[
        int, typer.Option(help="Outputs to sample into the queue (first time only).")
    ] = 80,
    port: Annotated[int, typer.Option()] = 7861,
) -> None:
    """Score a blind sample of outputs on the judge's rubric (local page) to validate the judge."""

    from legal_simplifier.eval.label_app import build_app
    from legal_simplifier.eval.labels import QUEUE_FILE, build_queue, label_dir

    if not (label_dir(run_dir) / QUEUE_FILE).exists():
        typer.echo(f"Sampled {len(build_queue(run_dir, n=n))} outputs into the label queue.")
    build_app(run_dir, reviewer).launch(server_port=port, share=False)


@app.command("judge-agreement")
def judge_agreement(
    run_dir: Annotated[Path, typer.Option("--run", help="An eval run directory.")],
    backend: Annotated[str, typer.Option(help="The judge to validate.")] = "claude",
    other: Annotated[
        str, typer.Option(help="Second judge to compare with, if it has judged the run.")
    ] = "hf",
) -> None:
    """Spearman and quadratic-weighted kappa (bootstrap CIs) between human labels and the judge."""

    from legal_simplifier.eval.agreement import describe
    from legal_simplifier.eval.agreement import judge_agreement as run_agreement

    try:
        result = run_agreement(run_dir, backend=backend, other=other)
    except FileNotFoundError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(describe(result))


@app.command("error-analysis")
def error_analysis(
    run_dir: Annotated[Path, typer.Option("--run", help="An eval run directory.")],
    samples: Annotated[int, typer.Option(help="Examples to show per failure type.")] = 3,
    config: EvalConfigOption = Path("configs/eval.yaml"),
) -> None:
    """Group failures into a taxonomy (detectors, metrics, low judge scores) with counts and examples."""

    from legal_simplifier.config import EvalConfigFile, load_config
    from legal_simplifier.eval.error_analysis import error_analysis as run_analysis

    low = load_config(config, EvalConfigFile).judge.low_score if config.exists() else 2
    result = run_analysis(run_dir, samples_per_type=samples, low_score=low)
    for name, per in result["counts"].items():
        if any(per.values()):
            typer.echo(f"{name}: " + ", ".join(f"{s} {n}" for s, n in per.items()))
    typer.echo(f"Wrote {run_dir / 'error_analysis.md'}")


@app.command("eval-gate")
def eval_gate(
    examples: Annotated[Path, typer.Option(help="Frozen outputs to re-score.")] = Path(
        "tests/fixtures/eval_gate/examples.jsonl"
    ),
    thresholds: Annotated[Path, typer.Option(help="Allowed ranges per metric.")] = Path(
        "tests/fixtures/eval_gate/thresholds.json"
    ),
) -> None:
    """CI gate: re-score stored outputs (mock fixture) and fail if a metric leaves its range."""

    from legal_simplifier.eval.gate import run_gate

    result = run_gate(examples, thresholds)
    for name, value in result.metrics.items():
        typer.echo(f"  {name}: {value:.3f}")
    if not result.passed:
        for failure in result.failures:
            typer.echo(f"FAIL {failure}", err=True)
        raise typer.Exit(code=1)
    typer.echo("Eval gate passed.")


@app.command("serve")
def serve() -> None:
    """Start the FastAPI inference service."""

    _not_yet("serve", 5)


@app.command("report")
def report(
    results_dir: Annotated[Path, typer.Option("--results-dir")] = Path("results"),
    run: Annotated[str | None, typer.Option(help="Primary run (default: the most recent).")] = None,
    readme: Annotated[
        Path, typer.Option(help="README to update between the results markers.")
    ] = Path("README.md"),
    figures_dir: Annotated[Path, typer.Option()] = Path("docs/figures"),
    training_log: Annotated[
        list[Path] | None,
        typer.Option(help="GRPO log_history.json or a Trackio/W&B CSV export (repeatable)."),
    ] = None,
    cost_log: Annotated[Path, typer.Option()] = Path("data/cache/anthropic/cost_log.jsonl"),
) -> None:
    """Build the README results table and docs/figures/. Refuses mock references and smoke runs."""

    from legal_simplifier.eval.guards import UnpublishableResultsError
    from legal_simplifier.eval.report import generate_report

    logs = training_log if training_log is not None else [Path("checkpoints/grpo/log_history.json")]
    try:
        result = generate_report(
            results_dir,
            run=run,
            readme=readme,
            figures_dir=figures_dir,
            training_logs=logs,
            cost_log=cost_log,
        )
    except UnpublishableResultsError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"Report from {result.run}: {result.readme}, {result.report_md}, {len(result.figures)} figures"
    )


if __name__ == "__main__":
    app()
