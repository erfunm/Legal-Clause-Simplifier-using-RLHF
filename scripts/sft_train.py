"""Supervised fine-tuning stub for the Legal Clause Simplifier project."""

from __future__ import annotations

import argparse
from pathlib import Path

from utils import ensure_directory


def parse_args() -> argparse.Namespace:
    """Return the CLI arguments expected by the SFT training script."""

    parser = argparse.ArgumentParser(description="Run baseline supervised fine-tuning")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sft.yaml"),
        help="Path to the YAML configuration file used for training.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/sft"),
        help="Directory where model checkpoints should be stored.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry-point for running supervised fine-tuning inside SageMaker."""

    args = parse_args()
    ensure_directory(args.output_dir)
    print(
        "[sft_train] Placeholder execution."
        f" Config: {args.config.resolve()} | Checkpoints will be saved under: {args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
