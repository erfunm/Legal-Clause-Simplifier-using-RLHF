"""Data preparation entry-point for the Legal Clause Simplifier project.

The real implementation will:
* Download the CUAD dataset inside SageMaker.
* Extract clause spans and generate simplified references.
* Persist processed data and metadata to ``data/processed`` and S3.

For now the script exposes a ``main`` stub so the CLI wiring is ready. This
allows subsequent steps to plug into SageMaker Processing jobs without
restructuring the repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from utils import ensure_directory


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the data preparation workflow."""

    parser = argparse.ArgumentParser(description="Prepare CUAD clauses for RLHF training")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Destination directory for processed artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    """Placeholder that ensures the output directory exists."""

    args = parse_args()
    ensure_directory(args.output_dir)
    print(f"[data_prep] Ready to populate processed data at: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
