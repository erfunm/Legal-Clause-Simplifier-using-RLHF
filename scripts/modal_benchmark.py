"""``lcs benchmark`` on one NVIDIA L4 with Modal (vLLM, falling back to transformers).

Needs the merged model and the test split on the Volume (``scripts/modal_rl.py``
writes ``checkpoints/grpo/merged``; ``data/processed`` was uploaded for SFT).

    modal run scripts/modal_benchmark.py                          # configs/benchmark.yaml, engine auto
    modal run scripts/modal_benchmark.py --engine transformers    # the same model without vLLM
    modal volume get legal-simplifier results/benchmarks results/benchmarks

The image mirrors ``docker/Dockerfile.gpu``: vLLM 0.31.0 with its own torch, plus
the package's serving extras (no training stack; ADR-028). The cost figure uses
``gpu_hourly_usd`` from the config (Modal's L4 GPU price, cited there), so it
covers the GPU only. The Claude comparison needs the eval run and the cost log
on the Volume too, or it falls back to a labelled Batch-price estimate.
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
REMOTE_REPO = "/root/project"
VOLUME_NAME = "legal-simplifier"
VOLUME_PATH = "/vol"
GPU = "L4"
TIMEOUT_S = 2 * 60 * 60
VLLM = "vllm==0.31.0"
IMAGE_FILES = ("pyproject.toml", "README.md")
IMAGE_DIRS = ("src", "configs")

image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(VLLM)
for name in IMAGE_FILES:
    image = image.add_local_file(REPO / name, f"{REMOTE_REPO}/{name}", copy=True)
for name in IMAGE_DIRS:
    image = image.add_local_dir(
        REPO / name, f"{REMOTE_REPO}/{name}", copy=True, ignore=["**/__pycache__"]
    )
image = (
    image.run_commands(f"python -m pip install --no-cache-dir '{REMOTE_REPO}[serve]'")
    .env({"HF_HOME": f"{VOLUME_PATH}/hf", "HF_HUB_DISABLE_TELEMETRY": "1"})
    .workdir(VOLUME_PATH)
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App("legal-simplifier-benchmark", image=image)


def run(config_yaml: str, engine: str = "", root: str | Path = VOLUME_PATH) -> str:
    import os

    import yaml

    from legal_simplifier.config import BenchmarkConfig
    from legal_simplifier.serving.benchmark import run_benchmark

    os.chdir(root)
    config = BenchmarkConfig.model_validate(yaml.safe_load(config_yaml))
    if engine:
        config.serve.engine = engine  # type: ignore[assignment]
        config.experiment.name = f"{config.experiment.name}-{engine}"
    model = Path(config.serve.model.name_or_path)
    if not (model / "config.json").exists():
        raise FileNotFoundError(
            f"{model} is not on the Volume. Train and merge first (scripts/modal_rl.py)."
        )
    if not config.clauses_path.exists():
        raise FileNotFoundError(
            f"{config.clauses_path} is not on the Volume. "
            f"Upload it: modal volume put {VOLUME_NAME} data/processed data/processed"
        )
    return str(Path(root) / run_benchmark(config))


@app.function(gpu=GPU, timeout=TIMEOUT_S, volumes={VOLUME_PATH: volume})
def benchmark(config_yaml: str, engine: str = "") -> str:
    try:
        return run(config_yaml, engine)
    finally:
        volume.commit()


@app.local_entrypoint()
def main(config: str = "configs/benchmark.yaml", engine: str = "") -> None:
    path = benchmark.remote((REPO / config).read_text(encoding="utf-8"), engine)
    print(f"Finished: {path} on Volume '{VOLUME_NAME}'.")
    print(f"Download: modal volume get {VOLUME_NAME} results/benchmarks results/benchmarks")
