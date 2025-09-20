"""Evaluation pipeline scaffold for the Legal Clause Simplifier project."""

from __future__ import annotations

import argparse
from pathlib import Path

from utils import ensure_directory


def parse_args() -> argparse.Namespace:
    """Arguments for running evaluation jobs on SageMaker."""

    parser = argparse.ArgumentParser(description="Evaluate baseline and PPO models")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory where metric summaries should be written.",
    )
    return parser.parse_args()


def main() -> None:
    """Ensure evaluation directories exist and report placeholder message."""

    args = parse_args()
    ensure_directory(args.output_dir)
    print(f"[evaluate] Metrics will be stored under: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
