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


@app.command("prepare-data")
def prepare_data(
    input_path: Annotated[Path, typer.Option("--input", help="Clause records (JSONL) to split.")],
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path("data/processed"),
    dedup_threshold: Annotated[
        float, typer.Option(help="Jaccard similarity treated as duplicate.")
    ] = 0.8,
    seed: Annotated[int, typer.Option()] = 13,
) -> None:
    """Deduplicate clause records and write contract-level train/dev/test splits."""

    from legal_simplifier.data.prepare import prepare_from_jsonl

    result = prepare_from_jsonl(input_path, output_dir, dedup_threshold=dedup_threshold, seed=seed)
    typer.echo(f"Wrote splits and stats to {result.output_dir}")


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
