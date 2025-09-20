"""PPO training scaffold using Hugging Face TRL."""

from __future__ import annotations

import argparse
from pathlib import Path

from utils import ensure_directory


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments expected by the PPO training job."""

    parser = argparse.ArgumentParser(description="Run PPO-based RLHF training")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ppo.yaml"),
        help="YAML file that stores PPO hyperparameters and generation settings.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/ppo"),
        help="Directory used to persist PPO checkpoints.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry-point used by SageMaker training jobs."""

    args = parse_args()
    ensure_directory(args.checkpoint_dir)
    print(
        "[ppo_train] Placeholder execution."
        f" Config: {args.config.resolve()} | Checkpoints: {args.checkpoint_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
