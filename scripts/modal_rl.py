"""GRPO, DPO and the GRPO ablations on NVIDIA L4s (24 GB) with Modal.

Needs the SFT run on the Volume first (``scripts/modal_sft.py`` writes
``checkpoints/sft/adapter``). See docs/RUNBOOK.md §3.

    modal run scripts/modal_rl.py --stage grpo                       # configs/grpo.yaml
    modal run scripts/modal_rl.py --stage dpo                        # build pairs, then DPO
    modal run scripts/modal_rl.py --stage ablations                  # every configs/ablations/*.yaml,
                                                                     # one L4 each, in parallel
    modal run scripts/modal_rl.py --stage ablations --only ablation-beta-0.04
    modal run scripts/modal_rl.py --stage grpo --wandb               # W&B (Modal secret 'wandb')
    modal volume get legal-simplifier checkpoints/grpo checkpoints/grpo

Same layout as the SFT script: one Volume at ``/vol`` holds data, checkpoints,
the HF cache (base model and the NLI reward model) and the local Trackio
store; config paths are relative to it. For GRPO and DPO the config is read
locally and sent with the call; ablation files are resolved inside the
container from the image's ``configs/``. Only an allowlist of repo files goes
into the image, and nothing is pushed to the Hub.

Memory on one L4 follows ADR-002: TRL's chunked log-prob head, ``beta: 0``
by default, gradient checkpointing, and ``memory/max_allocated_gb`` logged
every step so the first run records the real peak.
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
REMOTE_REPO = "/root/project"
VOLUME_NAME = "legal-simplifier"
VOLUME_PATH = "/vol"
GPU = "L4"
TIMEOUT_S = 12 * 60 * 60
TORCH = "torch==2.15.0"  # PyPI's Linux wheel bundles CUDA; the lockfile pins the CPU build for CI
IMAGE_FILES = ("pyproject.toml", "README.md")
IMAGE_DIRS = ("src", "configs")
STAGES = ("grpo", "dpo", "ablations")

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
app = modal.App("legal-simplifier-rl", image=image)


def _require(paths: list[Path], hint: str) -> None:
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"{missing[0]} is not on the Volume. {hint}")


def run_stage(
    stage: str,
    payload: str,
    root: str | Path = VOLUME_PATH,
    ablation_dir: str | Path = f"{REMOTE_REPO}/configs/ablations",
) -> str:
    """Run one stage from the Volume root; ``payload`` is a config YAML or an ablation name."""

    import os

    import yaml

    from legal_simplifier.config import DPOConfigFile, GRPOConfigFile
    from legal_simplifier.training.gpu import adapt_precision

    os.chdir(root)
    sft_hint = "Run scripts/modal_sft.py first."
    data_hint = f"Upload it: modal volume put {VOLUME_NAME} data/processed data/processed"
    if stage == "ablations":
        from legal_simplifier.training.ablations import discover

        (ablation,) = discover(ablation_dir, [payload])
        config = ablation.config
    elif stage == "grpo":
        config = GRPOConfigFile.model_validate(yaml.safe_load(payload))
    elif stage == "dpo":
        config = DPOConfigFile.model_validate(yaml.safe_load(payload))
    else:
        raise ValueError(f"unknown stage {stage!r}; choose from {STAGES}")

    _require([config.policy.sft_adapter / "adapter_config.json"], sft_hint)
    if config.tracking.backend == "wandb" and not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("tracking.backend is wandb: run with --wandb (Modal secret 'wandb')")
    config = adapt_precision(config)

    if isinstance(config, GRPOConfigFile):
        from legal_simplifier.training.grpo import train_grpo

        _require([config.data.train_path], data_hint)
        return str(Path(root) / train_grpo(config))

    from legal_simplifier.training.dpo import train_dpo
    from legal_simplifier.training.preferences import build_preferences

    _require([config.pairs.data_dir / f"{s}.jsonl" for s in config.pairs.splits], data_hint)
    build_preferences(config)  # kept on the Volume even if DPO then fails (commit in finally)
    return str(Path(root) / train_dpo(config))


@app.function(gpu=GPU, timeout=TIMEOUT_S, volumes={VOLUME_PATH: volume})
def train(stage: str, payload: str) -> str:
    try:
        return run_stage(stage, payload)
    finally:
        volume.commit()


@app.function(
    gpu=GPU,
    timeout=TIMEOUT_S,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("wandb")],
)
def train_with_wandb(stage: str, payload: str) -> str:
    try:
        return run_stage(stage, payload)
    finally:
        volume.commit()


def ablation_names(directory: Path, only: str = "") -> list[str]:
    """Names from the local ablation files (read as plain YAML; no package import needed)."""

    import yaml

    names = [
        yaml.safe_load(p.read_text(encoding="utf-8"))["name"]
        for p in sorted(directory.glob("*.yaml"))
    ]
    wanted = [n for n in only.split(",") if n]
    unknown = set(wanted) - set(names)
    if unknown:
        raise ValueError(f"unknown ablation(s): {sorted(unknown)}; known: {names}")
    return wanted or names


@app.local_entrypoint()
def main(stage: str = "grpo", config: str = "", only: str = "", wandb: bool = False) -> None:
    if stage not in STAGES:
        raise SystemExit(f"--stage must be one of {STAGES}")
    fn = train_with_wandb if wandb else train
    if stage == "ablations":
        names = ablation_names(REPO / "configs" / "ablations", only)
        print(f"Launching {len(names)} ablation runs in parallel: {', '.join(names)}")
        for run_dir in fn.starmap([("ablations", name) for name in names]):
            print(f"Finished: {run_dir}")
        return
    config_path = REPO / (config or f"configs/{stage}.yaml")
    run_dir = fn.remote(stage, config_path.read_text(encoding="utf-8"))
    print(f"Finished: {run_dir} on Volume '{VOLUME_NAME}'.")
    print(f"Download: modal volume get {VOLUME_NAME} {Path(run_dir).relative_to(VOLUME_PATH)} .")
