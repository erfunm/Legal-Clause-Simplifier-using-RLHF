"""SFT on one NVIDIA L4 (24 GB) with Modal.

One-time setup (see docs/RUNBOOK.md §2):

    uv tool install modal && modal setup
    modal volume create legal-simplifier
    modal volume put legal-simplifier data/processed data/processed
    modal secret create wandb WANDB_API_KEY=...        # only for tracking.backend: wandb

Run (the config is read locally and sent with the call, so edits need no rebuild):

    modal run scripts/modal_sft.py                                  # configs/sft.yaml
    modal run scripts/modal_sft.py --config configs/sft.yaml --wandb
    modal volume get legal-simplifier checkpoints/sft checkpoints/sft

Paths in the config are relative, and they resolve on the Volume: data from
``/vol/data/processed``, checkpoints to ``/vol/checkpoints``, with the
Hugging Face cache and the local Trackio store there too. Nothing is pushed to
the Hub. Only an allowlist of repo files goes into the image (``pyproject.toml``,
``README.md``, ``src/``, ``configs/``); local data and untracked files stay local.
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
REMOTE_REPO = "/root/project"
VOLUME_NAME = "legal-simplifier"
VOLUME_PATH = "/vol"
GPU = "L4"
TIMEOUT_S = 4 * 60 * 60
# PyPI's Linux torch wheel bundles CUDA; the repo's lockfile pins the CPU build for CI.
TORCH = "torch==2.15.0"
IMAGE_FILES = ("pyproject.toml", "README.md")
IMAGE_DIRS = ("src", "configs")

image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(TORCH)
for name in IMAGE_FILES:
    image = image.add_local_file(REPO / name, f"{REMOTE_REPO}/{name}", copy=True)
for name in IMAGE_DIRS:
    image = image.add_local_dir(
        REPO / name, f"{REMOTE_REPO}/{name}", copy=True, ignore=["**/__pycache__"]
    )
image = (
    image.run_commands(f"python -m pip install --no-cache-dir '{REMOTE_REPO}[train,tracking]'")
    .env(
        {
            "HF_HOME": f"{VOLUME_PATH}/hf",
            "TRACKIO_DIR": f"{VOLUME_PATH}/trackio",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
    )
    .workdir(VOLUME_PATH)
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App("legal-simplifier-sft", image=image)


def run_sft(config_yaml: str, root: str | Path = VOLUME_PATH) -> str:
    """Validate the config, check the data is on the Volume, train; return the run directory."""

    import os

    import yaml

    from legal_simplifier.config import SFTConfigFile
    from legal_simplifier.training.gpu import adapt_precision
    from legal_simplifier.training.sft import train_sft

    os.chdir(root)
    config = SFTConfigFile.model_validate(yaml.safe_load(config_yaml))
    missing = [p for p in (config.data.train_path,) if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            f"{missing[0]} is not on the Volume. Upload the processed data first: "
            f"modal volume put {VOLUME_NAME} data/processed data/processed"
        )
    if config.tracking.backend == "wandb" and not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("tracking.backend is wandb: run with --wandb (Modal secret 'wandb')")
    config = adapt_precision(config)
    return str(Path(root) / train_sft(config))


@app.function(gpu=GPU, timeout=TIMEOUT_S, volumes={VOLUME_PATH: volume})
def train(config_yaml: str) -> str:
    try:
        return run_sft(config_yaml)
    finally:
        volume.commit()


@app.function(
    gpu=GPU,
    timeout=TIMEOUT_S,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("wandb")],
)
def train_with_wandb(config_yaml: str) -> str:
    try:
        return run_sft(config_yaml)
    finally:
        volume.commit()


@app.local_entrypoint()
def main(config: str = "configs/sft.yaml", wandb: bool = False) -> None:
    config_yaml = (REPO / config).read_text(encoding="utf-8")
    run_dir = (train_with_wandb if wandb else train).remote(config_yaml)
    print(f"Finished: {run_dir} on Volume '{VOLUME_NAME}'.")
    print(f"Download: modal volume get {VOLUME_NAME} {Path(run_dir).relative_to(VOLUME_PATH)} .")
