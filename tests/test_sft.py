"""SFT with the tiny random Qwen3.5 model on CPU (same chat template and vocab as the real base)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from legal_simplifier.cli import app
from legal_simplifier.config import ModelConfig, SFTConfigFile
from legal_simplifier.data.prepare import prepare_from_records
from legal_simplifier.data.schema import ClauseRecord
from legal_simplifier.prompts import build_messages
from legal_simplifier.training.sft import build_sft_trainer, records_to_examples

from .conftest import REPO_ROOT, TINY_QWEN35, TINY_QWEN35_REVISION


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


def _tiny_config(data_dir: Path, output_dir: Path) -> SFTConfigFile:
    return SFTConfigFile.model_validate(
        {
            "experiment": {"name": "test-sft", "seed": 0},
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
            },
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
    config_path.write_text(json.dumps(config))  # JSON is valid YAML
    result = runner.invoke(app, ["train-sft", "--config", str(config_path)])
    assert result.exit_code == 0, result.output

    out = tmp_path / "out"
    assert (out / "model.safetensors").exists()
    assert (out / "tokenizer_config.json").exists()
    assert "train_loss" in json.loads((out / "metrics.json").read_text())
    assert json.loads((out / "experiment.json").read_text())["experiment"]["name"] == "test-sft"
