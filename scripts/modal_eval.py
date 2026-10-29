"""``lcs evaluate`` for the local systems (base, SFT, SFT+DPO, SFT+GRPO) on one NVIDIA L4.

Needs the trained adapters on the Volume (``scripts/modal_sft.py`` and
``scripts/modal_rl.py`` write them) and the golden set uploaded:

    modal volume put legal-simplifier data/golden data/golden

Run:

    modal run scripts/modal_eval.py                          # configs/eval.yaml, --skip-paid
    modal run scripts/modal_eval.py --judge-hf               # then the open-weight judge (Granite 8B)
    modal volume get legal-simplifier results/eval-v1 results/eval-v1

The paid parts (the Claude system and the Claude judge) need no GPU: run them
from your machine with ``--yes`` after downloading the run (docs/RUNBOOK.md §4).
Generations are cached on the Volume by system fingerprint, so re-running only
generates what changed. Same image allowlist and Volume layout as the other
Modal scripts; nothing is pushed to the Hub.
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
REMOTE_REPO = "/root/project"
VOLUME_NAME = "legal-simplifier"
VOLUME_PATH = "/vol"
GPU = "L4"
TIMEOUT_S = 6 * 60 * 60
TORCH = "torch==2.15.0"  # PyPI's Linux wheel bundles CUDA; the lockfile pins the CPU build for CI
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
    image.run_commands(f"python -m pip install --no-cache-dir '{REMOTE_REPO}[train,eval]'")
    .env({"HF_HOME": f"{VOLUME_PATH}/hf", "HF_HUB_DISABLE_TELEMETRY": "1"})
    .workdir(VOLUME_PATH)
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App("legal-simplifier-eval", image=image)


def run_eval(config_yaml: str, *, judge_hf: bool = False, root: str | Path = VOLUME_PATH) -> str:
    """Evaluate the free systems (and optionally run the HF judge) from the Volume root."""

    import os

    import yaml

    from legal_simplifier.config import EvalConfigFile
    from legal_simplifier.eval.evaluate import aggregate_run, load_run, run_evaluation

    os.chdir(root)
    config = EvalConfigFile.model_validate(yaml.safe_load(config_yaml))
    golden = config.data.golden_dir / "drafts.jsonl"
    if not golden.exists():
        raise FileNotFoundError(
            f"{golden} is not on the Volume. Upload it: modal volume put {VOLUME_NAME} data/golden data/golden"
        )
    missing = [
        str(s.adapter)
        for s in config.systems
        if s.adapter is not None and not (s.adapter / "adapter_config.json").exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"adapters not on the Volume: {missing}. Run scripts/modal_sft.py / modal_rl.py first."
        )
    run_dir = run_evaluation(config, skip_paid=True)
    if judge_hf:
        from legal_simplifier.eval.judge import HFJudge, build_tasks, write_judgements

        _, rows = load_run(run_dir)
        result = HFJudge(config.judge).judge(build_tasks(rows, seed=config.judge.seed))
        write_judgements(run_dir, "hf", result.rows)
        aggregate_run(run_dir, config)
    return str(Path(root) / run_dir)


@app.function(gpu=GPU, timeout=TIMEOUT_S, volumes={VOLUME_PATH: volume})
def evaluate(config_yaml: str, judge_hf: bool = False) -> str:
    try:
        return run_eval(config_yaml, judge_hf=judge_hf)
    finally:
        volume.commit()


@app.local_entrypoint()
def main(config: str = "configs/eval.yaml", judge_hf: bool = False) -> None:
    run_dir = evaluate.remote((REPO / config).read_text(encoding="utf-8"), judge_hf)
    print(f"Finished: {run_dir} on Volume '{VOLUME_NAME}'.")
    print(
        f"Download: modal volume get {VOLUME_NAME} {Path(run_dir).relative_to(VOLUME_PATH)} results/"
    )
