"""Fast checks of the training setup: tracking, QLoRA config, configs and model cards."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_simplifier.config import QuantizationConfig, SFTConfigFile, TrackingConfig, load_config
from legal_simplifier.training.lora import build_lora_config, build_quantization_config
from legal_simplifier.training.model_card import render_model_card
from legal_simplifier.training.tracking import TrackingPublishRefused, configure_tracking

from .conftest import REPO_ROOT

# --- tracking --------------------------------------------------------------------


def test_no_tracking() -> None:
    env: dict[str, str] = {}
    assert configure_tracking(TrackingConfig(), env) == {"report_to": ["none"]}
    assert env == {}


def test_wandb_project_and_mode_come_from_config_not_keys() -> None:
    env: dict[str, str] = {}
    kwargs = configure_tracking(
        TrackingConfig(backend="wandb", project="p", wandb_entity="team", wandb_mode="offline"),
        env,
    )
    assert kwargs["report_to"] == ["wandb"]
    assert env == {"WANDB_PROJECT": "p", "WANDB_MODE": "offline", "WANDB_ENTITY": "team"}
    assert "WANDB_API_KEY" not in env


def test_trackio_stays_local_by_default(tmp_path: Path) -> None:
    env: dict[str, str] = {}
    kwargs = configure_tracking(TrackingConfig(backend="trackio", trackio_dir=tmp_path), env)
    assert kwargs == {
        "report_to": ["trackio"],
        "project": "legal-clause-simplifier",
        "trackio_space_id": None,
    }
    assert env == {"TRACKIO_DIR": str(tmp_path)}


@pytest.mark.parametrize(
    "var", ["TRACKIO_SPACE_ID", "TRACKIO_SERVER_URL", "TRACKIO_REGISTRY_BUCKET_ID"]
)
def test_ambient_trackio_remote_is_refused(var: str) -> None:
    with pytest.raises(TrackingPublishRefused, match=var):
        configure_tracking(TrackingConfig(backend="trackio"), {var: "someone/space"})
    explicit = configure_tracking(
        TrackingConfig(backend="trackio", trackio_space_id="me/dash"), {var: "x"}
    )
    assert explicit["trackio_space_id"] == "me/dash"


# --- LoRA / QLoRA config -----------------------------------------------------------


def test_lora_config_is_built_from_the_yaml() -> None:
    cfg = load_config(REPO_ROOT / "configs" / "sft.yaml", SFTConfigFile)
    peft_config = build_lora_config(cfg.lora, base_revision=cfg.model.revision)
    assert peft_config.r == 16 and peft_config.lora_alpha == 32
    assert set(peft_config.target_modules) == set(cfg.lora.target_modules)  # type: ignore[arg-type]
    assert "in_proj_qkv" in cfg.lora.target_modules and "lm_head" not in cfg.lora.target_modules
    assert peft_config.revision == cfg.model.revision


def test_qlora_builds_a_4bit_config_without_a_gpu() -> None:
    import torch

    assert build_quantization_config(QuantizationConfig()) is None
    bnb = build_quantization_config(QuantizationConfig(load_in_4bit=True))
    assert bnb is not None and bnb.load_in_4bit
    assert bnb.bnb_4bit_quant_type == "nf4" and bnb.bnb_4bit_use_double_quant
    assert bnb.bnb_4bit_compute_dtype == torch.bfloat16


def test_qlora_requires_lora() -> None:
    with pytest.raises(ValueError, match="requires lora"):
        SFTConfigFile.model_validate(
            {
                "experiment": {"name": "x"},
                "lora": {"enabled": False},
                "quantization": {"load_in_4bit": True},
            }
        )


def test_real_config_is_not_a_smoke_run_and_does_not_publish() -> None:
    cfg = load_config(REPO_ROOT / "configs" / "sft.yaml", SFTConfigFile)
    assert not cfg.experiment.smoke
    assert cfg.tracking.trackio_space_id is None
    smoke = load_config(REPO_ROOT / "configs" / "smoke" / "sft.yaml", SFTConfigFile)
    assert smoke.experiment.smoke and smoke.training.use_cpu


# --- model card ----------------------------------------------------------------------


def _experiment(smoke: bool = False) -> dict:
    cfg = load_config(REPO_ROOT / "configs" / "sft.yaml", SFTConfigFile)
    cfg.experiment.smoke = smoke
    return json.loads(cfg.model_dump_json())


STATS = {
    "splits": {"train": {"clauses": 10}, "dev": {"clauses": 2}, "test": {"clauses": 2}},
    "reference_source": {"claude": 14},
    "references": {"model": "claude-sonnet-5-5"},
}


def test_card_for_a_real_run_shows_training_loss_but_no_eval_numbers() -> None:
    card = render_model_card(
        "adapter",
        _experiment(),
        lora={"parameters": {"trainable": 100, "total": 1000}},
        metrics={"train_loss": 1.2345, "train_runtime": 10.0},
        data_stats=STATS,
    )
    assert card.startswith("---\nbase_model: Qwen/Qwen3.5-0.8B\nbase_model_relation: adapter")
    assert "library_name: peft" in card and "license: apache-2.0" in card
    assert "`train_loss` 1.234" in card and "train_runtime" not in card
    assert "Results: pending first full run" in card
    assert "100 of 1,000" in card and "`in_proj_qkv`" in card
    assert "train 10, dev 2, test 2" in card and "claude-sonnet-5-5" in card
    assert "Smoke test" not in card and "must not be published" not in card
    assert "AutoPeftModelForCausalLM" in card


def test_card_for_merged_weights() -> None:
    card = render_model_card("merged", _experiment(), data_stats=STATS)
    assert "base_model_relation: finetune" in card and "library_name: transformers" in card
    assert "AutoModelForCausalLM.from_pretrained(path)" in card


def test_smoke_and_mock_runs_are_flagged_and_hide_metrics() -> None:
    card = render_model_card(
        "adapter",
        _experiment(smoke=True),
        metrics={"train_loss": 12.4},
        data_stats={**STATS, "reference_source": {"mock": 14}},
    )
    assert "Smoke test" in card and "mock references" in card
    assert "12.4" not in card
