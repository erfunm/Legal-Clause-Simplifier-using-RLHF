"""GRPO, preference pairs and DPO on CPU with the tiny random Qwen3.5 (slow tests).

The SFT adapter they start from is trained once per session with the same
code as ``lcs train-sft``. Reward scorers are fakes except in the NLI test,
which loads a tiny random DeBERTa classifier.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from legal_simplifier.cli import app
from legal_simplifier.config import DPOConfigFile, GRPOConfigFile, ModelConfig, SFTConfigFile
from legal_simplifier.data.prepare import prepare_from_records
from legal_simplifier.data.schema import read_records
from legal_simplifier.rewards.suite import REWARD_NAMES, RewardSuite
from legal_simplifier.training.dpo import build_dpo_trainer, train_dpo
from legal_simplifier.training.grpo import build_grpo_trainer, train_grpo
from legal_simplifier.training.preferences import build_preferences
from legal_simplifier.training.sft import ADAPTER_DIR, MERGED_DIR, train_sft

from .conftest import FIXTURES, TINY_QWEN35, TINY_QWEN35_REVISION

pytestmark = pytest.mark.slow

TINY_DEBERTA = "hf-internal-testing/tiny-random-DebertaV2ForSequenceClassification"
TINY_DEBERTA_REVISION = "035f03d5411169990d9ac33970a084ad98c0769a"
MODEL = {"name_or_path": TINY_QWEN35, "revision": TINY_QWEN35_REVISION, "dtype": "float32"}


class FakeNLI:
    def __init__(self) -> None:
        self.calls = 0

    def entailment(self, premises: Sequence[str], hypotheses: Sequence[str]) -> list[float]:
        self.calls += 1
        return [min(1.0, len(h) / 100) for h in hypotheses]


@pytest.fixture(scope="session")
def sft_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("rl")
    data_dir = root / "data"
    prepare_from_records(read_records(FIXTURES / "clauses.jsonl"), data_dir)
    config = SFTConfigFile.model_validate(
        {
            "experiment": {"name": "sft-for-rl", "seed": 0, "smoke": True},
            "data": {"train_path": data_dir / "train.jsonl", "dev_path": data_dir / "dev.jsonl"},
            "model": MODEL,
            "lora": {"r": 4, "alpha": 8, "dropout": 0.0},
            "training": {
                "output_dir": root / "sft",
                "max_length": 256,
                "per_device_train_batch_size": 2,
                "max_steps": 1,
                "eval_strategy": "no",
                "save_strategy": "no",
                "use_cpu": True,
                "save_merged": False,
            },
        }
    )
    run = train_sft(config)
    return {"root": root, "data": data_dir, "adapter": run / ADAPTER_DIR}


def _grpo_config(sft_run: dict[str, Path], out: Path, **grpo: Any) -> GRPOConfigFile:
    return GRPOConfigFile.model_validate(
        {
            "experiment": {"name": "test-grpo", "seed": 0, "smoke": True},
            "data": {"train_path": sft_run["data"] / "train.jsonl"},
            "model": MODEL,
            "policy": {"sft_adapter": sft_run["adapter"]},
            "grpo": {
                "output_dir": out,
                "num_generations": 2,
                "max_completion_length": 12,
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "max_steps": 2,
                "logging_steps": 1,
                "save_strategy": "no",
                "mask_truncated_completions": False,
                "log_completions": False,
                "use_cpu": True,
                "save_merged": False,
                "learning_rate": 1e-2,
                **grpo,
            },
        }
    )


def _lora_tensors(model: Any, adapter: str = "default") -> dict[str, Any]:
    return {
        n.replace(f".{adapter}.", "."): p.detach().clone()
        for n, p in model.named_parameters()
        if f".lora_A.{adapter}." in n or f".lora_B.{adapter}." in n
    }


def test_grpo_starts_from_the_sft_adapter(sft_run: dict[str, Path], tmp_path: Path) -> None:
    import torch
    from safetensors.torch import load_file

    config = _grpo_config(sft_run, tmp_path / "g")
    trainer = build_grpo_trainer(config, suite=RewardSuite(config.rewards, entailment=FakeNLI()))
    saved = load_file(str(sft_run["adapter"] / "adapter_model.safetensors"))
    live = _lora_tensors(trainer.model)
    assert live and len(live) == len(saved)
    for name, tensor in live.items():  # same names as in adapter_model.safetensors
        assert torch.equal(tensor, saved[name])
    trainable = [n for n, p in trainer.model.named_parameters() if p.requires_grad]
    assert trainable and all(".lora_" in n and ".default." in n for n in trainable)
    assert trainer.ref_model is None and "ref" not in trainer.model.peft_config  # beta 0


def test_grpo_logs_each_reward_separately_and_saves_the_run(
    sft_run: dict[str, Path], tmp_path: Path
) -> None:
    from peft import PeftModel

    from legal_simplifier.training.sft import load_base_model

    config = _grpo_config(sft_run, tmp_path / "grpo", save_merged=True)
    config.rewards.weights.entity_preservation = 0.0  # the entity ablation
    nli = FakeNLI()
    trainer = build_grpo_trainer(config, suite=RewardSuite(config.rewards, entailment=nli))
    assert trainer.args.reward_weights == [1.0, 1.0, 0.0, 1.0]
    run = train_grpo(config, suite=RewardSuite(config.rewards, entailment=nli))

    history = json.loads((run / "log_history.json").read_text())
    steps = [h for h in history if "loss" in h]
    assert len(steps) == 2
    for name in REWARD_NAMES:  # logged on its own, weight 0 included
        assert all(f"rewards/{name}/mean" in h and f"rewards/{name}/std" in h for h in steps)
    assert nli.calls >= 2

    experiment = json.loads((run / "experiment.json").read_text())
    assert (
        experiment["stage"] == "grpo"
        and experiment["rewards"]["weights"]["entity_preservation"] == 0.0
    )
    adapter_config = json.loads((run / ADAPTER_DIR / "adapter_config.json").read_text())
    assert adapter_config["base_model_name_or_path"] == TINY_QWEN35
    loaded = PeftModel.from_pretrained(
        load_base_model(ModelConfig(**MODEL)), str(run / ADAPTER_DIR)
    )
    assert _lora_tensors(loaded)
    assert (run / MERGED_DIR / "model.safetensors").exists()
    card = (run / ADAPTER_DIR / "README.md").read_text()
    assert "GRPO post-training" in card and "| num_generations | 2 |" in card
    assert "entity_preservation 0.0" in card and "Results: pending first full run" in card


def test_beta_above_zero_uses_a_frozen_copy_of_the_sft_adapter(
    sft_run: dict[str, Path], tmp_path: Path
) -> None:
    import torch

    config = _grpo_config(sft_run, tmp_path / "kl", beta=0.04)
    suite = RewardSuite(config.rewards, entailment=FakeNLI())
    trainer = build_grpo_trainer(config, suite=suite)
    assert "ref" in trainer.model.peft_config and trainer.ref_model is None  # no second base model
    ref, default = _lora_tensors(trainer.model, "ref"), _lora_tensors(trainer.model, "default")
    assert ref.keys() == default.keys() and all(torch.equal(ref[k], default[k]) for k in ref)
    assert not any(p.requires_grad for n, p in trainer.model.named_parameters() if ".ref." in n)

    run = train_grpo(config, suite=suite)
    history = [h for h in json.loads((run / "log_history.json").read_text()) if "loss" in h]
    assert all("kl" in h for h in history)
    assert not (run / ADAPTER_DIR / "ref").exists()  # only the trained adapter is saved


def test_build_preferences_then_dpo(sft_run: dict[str, Path], tmp_path: Path) -> None:
    config = DPOConfigFile.model_validate(
        {
            "experiment": {"name": "test-dpo", "seed": 0, "smoke": True},
            "model": MODEL,
            "policy": {"sft_adapter": sft_run["adapter"]},
            "pairs": {
                "data_dir": sft_run["data"],
                "output_dir": tmp_path / "prefs",
                "n_samples": 3,
                "max_new_tokens": 10,
                "batch_size": 2,
                "min_margin": 0.0,
                "require_clean_chosen": False,
            },
            "dpo": {
                "output_dir": tmp_path / "dpo",
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "max_steps": 2,
                "max_length": 256,
                "logging_steps": 1,
                "save_strategy": "no",
                "eval_strategy": "no",
                "use_cpu": True,
                "save_merged": False,
            },
        }
    )
    stats = build_preferences(config, suite=RewardSuite(config.rewards, entailment=FakeNLI()))
    assert set(stats) == {"train", "dev"}
    n_train = len(read_records(sft_run["data"] / "train.jsonl"))
    samples = [
        json.loads(line)
        for line in (tmp_path / "prefs" / "samples_train.jsonl").read_text().splitlines()
    ]
    assert len(samples) == n_train and all(len(s["outputs"]) == 3 for s in samples)
    pairs = [
        json.loads(line) for line in (tmp_path / "prefs" / "train.jsonl").read_text().splitlines()
    ]
    assert len(pairs) == stats["train"].n_pairs > 0
    for pair in pairs:
        assert pair["chosen_score"]["total"] >= pair["rejected_score"]["total"]
        assert pair["chosen"][0]["role"] == "assistant" and pair["prompt"][0]["role"] == "system"
    assert json.loads((tmp_path / "prefs" / "stats.json").read_text())["splits"]["train"][
        "n_pairs"
    ] == len(pairs)

    trainer = build_dpo_trainer(config)
    assert "ref" in trainer.model.peft_config  # frozen SFT copy as the DPO reference
    run = train_dpo(config)
    history = [h for h in json.loads((run / "log_history.json").read_text()) if "loss" in h]
    assert len(history) == 2 and all("rewards/margins" in h for h in history)
    assert (run / ADAPTER_DIR / "adapter_model.safetensors").exists()
    assert not (run / ADAPTER_DIR / "ref").exists()
    assert json.loads((run / "experiment.json").read_text())["stage"] == "dpo"
    assert "DPO post-training" in (run / ADAPTER_DIR / "README.md").read_text()

    # The stage-aware merge works on an RL run directory too.
    result = CliRunner().invoke(app, ["merge-adapter", "--run-dir", str(run)])
    assert result.exit_code == 0, result.output
    assert (run / MERGED_DIR / "model.safetensors").exists()


def test_dpo_without_pairs_explains_what_to_run(sft_run: dict[str, Path], tmp_path: Path) -> None:
    config = DPOConfigFile.model_validate(
        {
            "experiment": {"name": "x"},
            "model": MODEL,
            "policy": {"sft_adapter": sft_run["adapter"]},
            "pairs": {"output_dir": tmp_path / "empty"},
        }
    )
    with pytest.raises(ValueError, match="lcs build-preferences"):
        build_dpo_trainer(config)


def test_nli_scorer_with_a_tiny_classifier() -> None:
    from legal_simplifier.rewards.components import NLIScorer

    scorer = NLIScorer(
        TINY_DEBERTA,
        revision=TINY_DEBERTA_REVISION,
        entailment_label="LABEL_1",
        batch_size=2,
        max_length=64,
    )
    premises = ["The Buyer shall pay the price within thirty days. " * 20, "b", "c"]
    scores = scorer.entailment(premises, ["You must pay.", "b", "c"])
    assert len(scores) == 3 and all(0.0 <= s <= 1.0 for s in scores)
    assert scores == pytest.approx(scorer.entailment(premises, ["You must pay.", "b", "c"]))
    with pytest.raises(ValueError, match=r"set meaning\.entailment_label"):
        NLIScorer(TINY_DEBERTA, revision=TINY_DEBERTA_REVISION)
    with pytest.raises(ValueError, match="has no label"):
        NLIScorer(TINY_DEBERTA, revision=TINY_DEBERTA_REVISION, entailment_label="entailment")
