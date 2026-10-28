"""``lcs`` command-line interface."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

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
    """Supervised fine-tuning with TRL SFTTrainer (completion-only loss)."""

    from legal_simplifier.config import SFTConfigFile, load_config
    from legal_simplifier.training.sft import train_sft as run

    cfg = load_config(config, SFTConfigFile)
    if output_dir is not None:
        cfg.training.output_dir = output_dir
    typer.echo(f"Saved model to {run(cfg)}")


@app.command("train-grpo")
def train_grpo() -> None:
    """GRPO from the SFT checkpoint with split reward functions."""

    _not_yet("train-grpo", 3)


@app.command("train-dpo")
def train_dpo() -> None:
    """DPO on preference pairs ranked by the composite reward."""

    _not_yet("train-dpo", 3)


@app.command("evaluate")
def evaluate() -> None:
    """Run every system on the eval sets and write results/<run>/."""

    _not_yet("evaluate", 4)


@app.command("serve")
def serve() -> None:
    """Start the FastAPI inference service."""

    _not_yet("serve", 5)


@app.command("report")
def report(
    results_dir: Annotated[Path, typer.Option("--results-dir")] = Path("results"),
) -> None:
    """Build results tables and figures. Refuses mock references and smoke runs."""

    from legal_simplifier.eval.guards import UnpublishableResultsError
    from legal_simplifier.eval.report import generate_report

    try:
        generate_report(results_dir)
    except UnpublishableResultsError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except NotImplementedError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc


if __name__ == "__main__":
    app()
