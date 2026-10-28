"""Supervised fine-tuning with TRL's ``SFTTrainer``.

Each record becomes a conversational prompt/completion pair, which the model's
chat template renders. ``completion_only_loss=True`` makes TRL set the labels
of every prompt token to -100, so only the simplification is learned. (The old
script masked the prompt and then used a collator that rebuilt the labels from
``input_ids``, which put the loss back on the prompt.)
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from legal_simplifier.config import SFTConfigFile
from legal_simplifier.data.schema import ClauseRecord, read_records
from legal_simplifier.prompts import build_completion, build_messages

if TYPE_CHECKING:
    from datasets import Dataset
    from trl import SFTTrainer

LOGGER = logging.getLogger(__name__)


def records_to_examples(
    records: Sequence[ClauseRecord],
    *,
    include_category: bool = True,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Convert records to TRL conversational prompt-completion examples."""

    examples: list[dict[str, Any]] = []
    for record in records:
        example: dict[str, Any] = {
            "prompt": build_messages(record.clause, record.category if include_category else None),
            "completion": build_completion(record.reference),
        }
        if chat_template_kwargs:
            # TRL forwards this column to ``apply_chat_template``.
            example["chat_template_kwargs"] = chat_template_kwargs
        examples.append(example)
    return examples


def _to_dataset(examples: list[dict[str, Any]]) -> Dataset:
    from datasets import Dataset

    return Dataset.from_list(examples)


def build_sft_trainer(config: SFTConfigFile) -> SFTTrainer:
    """Create (but do not run) the trainer described by ``config``."""

    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    train_records = read_records(config.data.train_path)
    dev_path = config.data.dev_path
    dev_records = read_records(dev_path) if dev_path and Path(dev_path).exists() else []
    if not train_records:
        raise ValueError(f"no training records in {config.data.train_path}")

    convert = {
        "include_category": config.data.include_category,
        "chat_template_kwargs": config.model.chat_template_kwargs,
    }
    train_dataset = _to_dataset(records_to_examples(train_records, **convert))
    eval_dataset = _to_dataset(records_to_examples(dev_records, **convert)) if dev_records else None

    t = config.training
    args = SFTConfig(
        output_dir=str(t.output_dir),
        seed=config.experiment.seed,
        max_length=t.max_length,
        completion_only_loss=True,
        per_device_train_batch_size=t.per_device_train_batch_size,
        per_device_eval_batch_size=t.per_device_eval_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        num_train_epochs=t.num_train_epochs,
        max_steps=t.max_steps,
        learning_rate=t.learning_rate,
        weight_decay=t.weight_decay,
        warmup_steps=t.warmup_steps,
        lr_scheduler_type=t.lr_scheduler_type,
        bf16=t.bf16,
        fp16=t.fp16,
        gradient_checkpointing=t.gradient_checkpointing,
        logging_steps=t.logging_steps,
        eval_strategy=t.eval_strategy if eval_dataset is not None else "no",
        eval_steps=t.eval_steps,
        save_strategy=t.save_strategy,
        save_total_limit=t.save_total_limit,
        report_to=t.report_to,
        use_cpu=t.use_cpu,
        run_name=config.experiment.name,
        model_init_kwargs={"revision": config.model.revision, "dtype": config.model.dtype},
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name_or_path, revision=config.model.revision
    )
    return SFTTrainer(
        model=config.model.name_or_path,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )


def train_sft(config: SFTConfigFile) -> Path:
    """Run SFT, save the model, tokenizer and metrics, and return the output directory."""

    trainer = build_sft_trainer(config)
    LOGGER.info("Starting SFT for %s", config.experiment.name)
    result = trainer.train()
    metrics: dict[str, Any] = dict(result.metrics)
    if trainer.eval_dataset is not None:
        metrics.update(trainer.evaluate())

    output_dir = Path(config.training.output_dir)
    trainer.save_model(str(output_dir))
    trainer.processing_class.save_pretrained(str(output_dir))  # type: ignore[union-attr]
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    (output_dir / "experiment.json").write_text(config.model_dump_json(indent=2))
    LOGGER.info("Saved SFT model to %s", output_dir)
    return output_dir
