"""SFT with the tiny random Qwen3.5 model on CPU (same chat template and vocab as the real base)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from legal_simplifier.cli import app
from legal_simplifier.config import ModelConfig, SFTConfigFile
from legal_simplifier.data.prepare import prepare_from_records
from legal_simplifier.data.schema import ClauseRecord
from legal_simplifier.prompts import build_messages
from legal_simplifier.training.lora import check_lora_coverage
from legal_simplifier.training.sft import (
    ADAPTER_DIR,
    MERGED_DIR,
    build_sft_trainer,
    load_base_model,
    records_to_examples,
    train_sft,
)

from .conftest import REPO_ROOT, TINY_QWEN35, TINY_QWEN35_REVISION

if TYPE_CHECKING:
    from transformers import PreTrainedModel


def test_records_become_conversational_prompt_completion_pairs(
    fixture_records: list[ClauseRecord],
) -> None:
    record = fixture_records[0]
    (example,) = records_to_examples([record], chat_template_kwargs={"enable_thinking": False})
    assert example["prompt"] == build_messages(record.clause, record.category)
    assert example["completion"] == [{"role": "assistant", "content": record.reference}]
    assert example["chat_template_kwargs"] == {"enable_thinking": False}
    (no_kwargs,) = records_to_examples([record], include_category=False)
    assert "chat_template_kwargs" not in no_kwargs
    assert "Clause type" not in no_kwargs["prompt"][1]["content"]


def _tiny_config(data_dir: Path, output_dir: Path, **training: object) -> SFTConfigFile:
    return SFTConfigFile.model_validate(
        {
            "experiment": {"name": "test-sft", "seed": 0, "smoke": True},
            "data": {"train_path": data_dir / "train.jsonl", "dev_path": data_dir / "dev.jsonl"},
            "model": ModelConfig(
                name_or_path=TINY_QWEN35, revision=TINY_QWEN35_REVISION, dtype="float32"
            ).model_dump(),
            "training": {
                "output_dir": output_dir,
                "max_length": 256,
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "max_steps": 1,
                "eval_strategy": "no",
                "save_strategy": "no",
                "use_cpu": True,
                **training,
            },
            "lora": {"r": 4, "alpha": 8, "dropout": 0.0},
        }
    )


@pytest.mark.slow
def test_prompt_tokens_are_masked_and_completion_tokens_are_learned(
    tmp_path: Path, fixture_records: list[ClauseRecord]
) -> None:
    prepare_from_records(fixture_records, tmp_path / "data")
    trainer = build_sft_trainer(_tiny_config(tmp_path / "data", tmp_path / "out"))
    tokenizer = trainer.processing_class
    assert trainer.train_dataset is not None

    rows = [trainer.train_dataset[i] for i in range(2)]
    batch = trainer.data_collator(rows)

    for row_index in range(len(rows)):
        input_ids = batch["input_ids"][row_index]
        labels = batch["labels"][row_index]
        attention = batch["attention_mask"][row_index].bool()
        ids = input_ids[attention].tolist()
        labs = labels[attention].tolist()

        # The prompt is everything up to the first learned token.
        first_learned = next(i for i, lab in enumerate(labs) if lab != -100)
        assert first_learned > 0
        assert all(lab == -100 for lab in labs[:first_learned])
        assert labs[first_learned:] == ids[first_learned:]
        assert all(lab == -100 for lab in labels[~attention].tolist())

        prompt_text = tokenizer.decode(ids[:first_learned])  # type: ignore[union-attr]
        completion_text = tokenizer.decode(ids[first_learned:])  # type: ignore[union-attr]

        # The masked span is exactly the chat-template prompt (incl. the empty think block)
        # and the learned span is exactly the reference plus the end-of-turn token.
        (record,) = [r for r in fixture_records if " ".join(r.clause.split()) in prompt_text]
        expected_prompt = tokenizer.apply_chat_template(  # type: ignore[union-attr]
            build_messages(record.clause, record.category),
            tokenize=False,
            add_generation_prompt=True,
        )
        assert prompt_text == expected_prompt
        assert completion_text.startswith(record.reference)
        assert completion_text.rstrip().endswith("<|im_end|>")


# --- LoRA -----------------------------------------------------------------------

# Every projection of the two Qwen3.5 layer types that should carry an adapter.
EXPECTED_WRAPPED = {
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.out_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
}
EXPECTED_LEFT_OUT = {"lm_head", "linear_attn.in_proj_a", "linear_attn.in_proj_b"}


def _suffix(name: str) -> str:
    parts = name.split(".")
    return "lm_head" if parts[-1] == "lm_head" else ".".join(parts[-2:])


def _tiny_model() -> PreTrainedModel:
    return load_base_model(
        ModelConfig(name_or_path=TINY_QWEN35, revision=TINY_QWEN35_REVISION, dtype="float32")
    )


@pytest.mark.slow
def test_lora_wraps_every_projection_of_both_layer_types(
    tmp_path: Path, fixture_records: list[ClauseRecord]
) -> None:
    import torch
    from peft import PeftModel

    model = _tiny_model()
    assert type(model).__name__ == "Qwen3_5ForCausalLM"  # text-only, no vision tower
    assert set(model.config.layer_types) == {"linear_attention", "full_attention"}

    prepare_from_records(fixture_records, tmp_path / "data")
    trainer = build_sft_trainer(_tiny_config(tmp_path / "data", tmp_path / "out"), model=model)
    peft_model = trainer.model
    assert isinstance(peft_model, PeftModel)

    wrapped = {
        _suffix(name.removesuffix(".lora_A"))
        for name, _ in peft_model.named_modules()
        if name.endswith(".lora_A")
    }
    assert wrapped == EXPECTED_WRAPPED
    # Every layer, not just one of each type, carries the adapter.
    n_layers = model.config.num_hidden_layers
    per_layer = {
        int(name.split(".layers.")[1].split(".")[0])
        for name, _ in peft_model.named_modules()
        if name.endswith(".lora_A")
    }
    assert per_layer == set(range(n_layers))

    unwrapped_linear = {
        _suffix(name)
        for name, module in peft_model.named_modules()
        if type(module) is torch.nn.Linear
        and "lora_" not in name
        and not name.endswith(".base_layer")
    }
    assert unwrapped_linear == EXPECTED_LEFT_OUT

    trainable = {n for n, p in peft_model.named_parameters() if p.requires_grad}
    assert trainable and all(".lora_A." in n or ".lora_B." in n for n in trainable)
    assert not any("embed_tokens" in n or "lm_head" in n or "conv1d" in n for n in trainable)


@pytest.mark.slow
def test_a_target_that_matches_nothing_is_an_error() -> None:
    with pytest.raises(ValueError, match="in_proj_qkvz"):
        check_lora_coverage(_tiny_model(), ["q_proj", "in_proj_qkvz"])


@pytest.mark.slow
def test_two_step_run_saves_an_adapter_that_loads(
    tmp_path: Path, fixture_records: list[ClauseRecord]
) -> None:
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    prepare_from_records(fixture_records, tmp_path / "data")
    run_dir = train_sft(
        _tiny_config(tmp_path / "data", tmp_path / "run", max_steps=2, learning_rate=1e-2)
    )

    adapter_dir = run_dir / ADAPTER_DIR
    adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text())
    assert adapter_config["base_model_name_or_path"] == TINY_QWEN35
    assert adapter_config["revision"] == TINY_QWEN35_REVISION
    assert set(adapter_config["target_modules"]) == {s.split(".")[-1] for s in EXPECTED_WRAPPED}
    assert (adapter_dir / "tokenizer_config.json").exists()
    assert not (adapter_dir / "model.safetensors").exists()  # adapter only, not full weights

    saved = load_file(str(adapter_dir / "adapter_model.safetensors"))
    assert saved and all("lora_A" in k or "lora_B" in k for k in saved)
    assert any(t.abs().sum() > 0 for k, t in saved.items() if "lora_B" in k)  # it trained

    base = _tiny_model().eval()
    loaded = PeftModel.from_pretrained(_tiny_model(), str(adapter_dir)).eval()
    for name, tensor in loaded.state_dict().items():
        if "lora_" in name:
            key = name.replace(".default", "")
            assert torch.equal(tensor, saved[key])

    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    with torch.no_grad():
        base_logits = base(ids).logits
        adapter_logits = loaded(ids).logits
        merged = AutoModelForCausalLM.from_pretrained(str(run_dir / MERGED_DIR)).eval()
        merged_logits = merged(ids).logits
    assert not torch.allclose(base_logits, adapter_logits)
    assert torch.allclose(merged_logits, adapter_logits, atol=1e-4)

    lora = json.loads((run_dir / "lora.json").read_text())
    assert lora["coverage"]["unmatched_targets"] == []
    assert lora["parameters"]["trainable"] < lora["parameters"]["total"]
    assert "train_loss" in json.loads((run_dir / "metrics.json").read_text())

    for kind in (ADAPTER_DIR, MERGED_DIR):
        card = (run_dir / kind / "README.md").read_text()
        assert f"base_model: {TINY_QWEN35}" in card
        assert "Smoke test" in card and "Results: pending first full run" in card


@pytest.mark.slow
def test_smoke_pipeline_prepare_then_sft_via_cli(tmp_path: Path) -> None:
    runner = CliRunner()
    data_dir = tmp_path / "data"
    result = runner.invoke(
        app,
        [
            "prepare-data",
            "--input",
            str(REPO_ROOT / "tests/fixtures/clauses.jsonl"),
            "--output-dir",
            str(data_dir),
        ],
    )
    assert result.exit_code == 0, result.output

    config_path = tmp_path / "sft.yaml"
    config = _tiny_config(data_dir, tmp_path / "out").model_dump(mode="json")
    config["training"]["save_merged"] = False
    config_path.write_text(json.dumps(config))  # JSON is valid YAML
    result = runner.invoke(app, ["train-sft", "--config", str(config_path)])
    assert result.exit_code == 0, result.output

    out = tmp_path / "out"
    assert (out / ADAPTER_DIR / "adapter_model.safetensors").exists()
    assert not (out / MERGED_DIR).exists()
    assert json.loads((out / "experiment.json").read_text())["experiment"]["name"] == "test-sft"

    result = runner.invoke(app, ["merge-adapter", "--run-dir", str(out)])
    assert result.exit_code == 0, result.output
    assert (out / MERGED_DIR / "model.safetensors").exists()
    assert (out / MERGED_DIR / "README.md").exists()


@pytest.mark.slow
def test_full_fine_tune_still_works_without_lora(
    tmp_path: Path, fixture_records: list[ClauseRecord]
) -> None:
    prepare_from_records(fixture_records, tmp_path / "data")
    config = _tiny_config(tmp_path / "data", tmp_path / "run")
    config.lora.enabled = False
    run_dir = train_sft(config)
    assert (run_dir / "model" / "model.safetensors").exists()
    assert not (run_dir / ADAPTER_DIR).exists()
    assert "(LoRA adapter)" not in (run_dir / "model" / "README.md").read_text()
