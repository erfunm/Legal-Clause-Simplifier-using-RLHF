"""Supervised fine-tuning entry-point for the Legal Clause Simplifier project."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from utils import (
    ensure_directory,
    format_prompt,
    set_seed,
    upload_directory_to_s3,
)


LOGGER = logging.getLogger("sft_train")


def parse_args() -> argparse.Namespace:
    """Return the CLI arguments expected by the SFT training script."""

    parser = argparse.ArgumentParser(description="Run baseline supervised fine-tuning")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sft.yaml"),
        help="Path to the YAML configuration file used for training.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to override the config output path.",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    """Load the YAML configuration used for supervised fine-tuning."""

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _require_paths(paths: Mapping[str, Path | str]) -> None:
    missing = [name for name, value in paths.items() if value and not Path(value).exists()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"Missing dataset files required for training: {joined}")


def _prepare_datasets(
    raw_datasets: DatasetDict,
    tokenizer,
    text_column: str,
    target_column: str,
    prompt_template: str,
    max_source_tokens: int,
    max_target_tokens: int,
) -> DatasetDict:
    """Tokenise prompts and labels for causal language modelling."""

    max_length = max_source_tokens + max_target_tokens

    def _format_example(example: Mapping[str, Any]) -> Dict[str, Any]:
        clause = str(example[text_column]).strip()
        target = str(example[target_column]).strip()
        prompt = format_prompt(clause, template=prompt_template)
        full_text = f"{prompt} {target}"
        if tokenizer.eos_token:
            full_text = full_text.rstrip() + tokenizer.eos_token

        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        prompt_token_ids = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            padding=False,
            add_special_tokens=False,
        )["input_ids"]
        labels = tokenized["input_ids"].copy()
        prompt_length = min(len(labels), len(prompt_token_ids))
        for index in range(prompt_length):
            labels[index] = -100
        tokenized["labels"] = labels
        return tokenized

    processed = raw_datasets.map(
        _format_example,
        remove_columns=list(raw_datasets["train"].column_names),
        desc="Tokenising clauses",
    )
    return processed


def _build_training_arguments(
    output_dir: Path,
    training_cfg: Mapping[str, Any],
    report_to: Iterable[str] | None,
) -> TrainingArguments:
    logging_dir = output_dir / "runs"
    ensure_directory(logging_dir)
    return TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=training_cfg.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 1),
        num_train_epochs=training_cfg.get("num_train_epochs", 1),
        max_steps=training_cfg.get("max_train_steps"),
        learning_rate=training_cfg.get("learning_rate", 5e-5),
        weight_decay=training_cfg.get("weight_decay", 0.0),
        warmup_ratio=training_cfg.get("warmup_ratio", 0.0),
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "linear"),
        fp16=training_cfg.get("fp16", False),
        bf16=training_cfg.get("bf16", False),
        logging_steps=training_cfg.get("logging_steps", 100),
        save_steps=training_cfg.get("save_steps", 500),
        eval_steps=training_cfg.get("eval_steps", 500),
        evaluation_strategy=training_cfg.get("evaluation_strategy", "no"),
        save_total_limit=training_cfg.get("save_total_limit"),
        dataloader_num_workers=training_cfg.get("dataloader_num_workers", 0),
        max_grad_norm=training_cfg.get("max_grad_norm", 1.0),
        gradient_checkpointing=training_cfg.get("gradient_checkpointing", False),
        report_to=list(report_to) if report_to else None,
        logging_dir=str(logging_dir),
    )


def _serialise_metrics(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    serialisable: Dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            serialisable[key] = value
        else:
            try:
                serialisable[key] = float(value)
            except (TypeError, ValueError):
                serialisable[key] = str(value)
    return serialisable


def main() -> None:
    """Entry-point for running supervised fine-tuning inside SageMaker."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = parse_args()
    config = load_config(args.config)

    experiment_cfg = config.get("experiment", {})
    paths_cfg = config.get("paths", {})
    prompt_cfg = config.get("prompt", {})
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    monitoring_cfg = config.get("monitoring", {})

    if args.output_dir is not None:
        paths_cfg["output_dir"] = str(args.output_dir)

    seed = experiment_cfg.get("seed")
    if seed is not None:
        set_seed(int(seed))

    train_csv = Path(paths_cfg.get("train_csv", ""))
    eval_csv = Path(paths_cfg.get("eval_csv", ""))
    _require_paths({"train_csv": train_csv, "eval_csv": eval_csv})

    cache_dir_value = paths_cfg.get("cache_dir") or "data/raw/hf_cache"
    cache_dir = ensure_directory(cache_dir_value)
    output_dir_value = paths_cfg.get("output_dir") or "checkpoints/sft"
    output_dir = ensure_directory(output_dir_value)

    if not monitoring_cfg.get("use_wandb", False):
        os.environ.setdefault("WANDB_DISABLED", "true")

    LOGGER.info("Loading tokenizer %s", model_cfg.get("tokenizer_name", model_cfg.get("name_or_path")))
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.get("tokenizer_name", model_cfg.get("name_or_path")),
        cache_dir=str(cache_dir),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model_cfg.get("padding_side"):
        tokenizer.padding_side = model_cfg["padding_side"]

    LOGGER.info("Loading model %s", model_cfg.get("name_or_path"))
    if model_cfg.get("use_peft"):
        raise NotImplementedError("PEFT fine-tuning is not implemented in this baseline script.")
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.get("name_or_path"),
        cache_dir=str(cache_dir),
    )
    model.resize_token_embeddings(len(tokenizer))
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    data_files = {"train": str(train_csv), "validation": str(eval_csv)}
    LOGGER.info("Loading datasets from %s", data_files)
    raw_datasets = load_dataset("csv", data_files=data_files, cache_dir=str(cache_dir))
    text_column = paths_cfg.get("text_column", "clause")
    target_column = paths_cfg.get("target_column", "reference_simple")
    prompt_template = prompt_cfg.get("template") or "{clause}"
    max_source_tokens = int(prompt_cfg.get("max_source_tokens", 512))
    max_target_tokens = int(prompt_cfg.get("max_target_tokens", 196))

    tokenized_datasets = _prepare_datasets(
        raw_datasets,
        tokenizer,
        text_column=text_column,
        target_column=target_column,
        prompt_template=prompt_template,
        max_source_tokens=max_source_tokens,
        max_target_tokens=max_target_tokens,
    )

    for split, dataset in tokenized_datasets.items():
        LOGGER.info("Prepared %d samples for %s", len(dataset), split)

    report_to = monitoring_cfg.get("report_to")
    training_args = _build_training_arguments(output_dir, training_cfg, report_to)

    if training_args.gradient_checkpointing:
        model.config.use_cache = False

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    eval_dataset = tokenized_datasets.get("validation")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    LOGGER.info("Starting supervised fine-tuning")
    train_result = trainer.train()
    metrics = dict(train_result.metrics)

    if eval_dataset is not None:
        LOGGER.info("Training complete, evaluating on validation split")
        eval_metrics = trainer.evaluate()
        metrics.update({f"eval_{k}": v for k, v in eval_metrics.items()})
    else:
        LOGGER.warning("Validation dataset not provided; skipping evaluation stage")

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(_serialise_metrics(metrics), handle, indent=2)
    LOGGER.info("Metrics written to %s", metrics_path)

    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    output_s3_uri = paths_cfg.get("output_s3_uri")
    if output_s3_uri:
        try:
            upload_directory_to_s3(output_dir, output_s3_uri)
        except Exception as exc:  # pragma: no cover - network operations
            LOGGER.error("Failed to upload checkpoints to %s: %s", output_s3_uri, exc)

    tensorboard_uri = paths_cfg.get("tensorboard_s3_uri") or monitoring_cfg.get("tensorboard_s3_uri")
    tensorboard_dir = Path(training_args.logging_dir)
    if tensorboard_uri and tensorboard_dir.exists():
        try:
            upload_directory_to_s3(tensorboard_dir, tensorboard_uri)
        except Exception as exc:  # pragma: no cover - network operations
            LOGGER.error("Failed to upload TensorBoard logs to %s: %s", tensorboard_uri, exc)


if __name__ == "__main__":
    main()
