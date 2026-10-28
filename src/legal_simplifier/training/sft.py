"""Supervised fine-tuning with TRL's ``SFTTrainer`` and a PEFT LoRA (or QLoRA) adapter.

Each record becomes a conversational prompt/completion pair, which the model's
chat template renders. ``completion_only_loss=True`` makes TRL set the labels
of every prompt token to -100, so only the simplification is learned. (The old
script masked the prompt and then used a collator that rebuilt the labels from
``input_ids``, which put the loss back on the prompt.)

The base model is loaded here with ``AutoModelForCausalLM`` rather than by
TRL from the model id: Qwen3.5 checkpoints declare
``Qwen3_5ForConditionalGeneration``, which TRL would load with its vision
tower. This task is text only, so the text-only causal LM is trained.

Output layout under ``training.output_dir``:

* ``adapter/``: the LoRA adapter, tokenizer and model card (``README.md``).
* ``merged/``: base + adapter merged into plain weights, tokenizer and card
  (if ``training.save_merged``). Serving and the Phase 3 trainers load this.
* ``model/``: the full weights instead, when ``lora.enabled`` is false.
* ``metrics.json``, ``experiment.json`` and ``lora.json`` (wrapped layers and
  parameter counts).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from legal_simplifier.config import ModelConfig, QuantizationConfig, SFTConfigFile
from legal_simplifier.data.schema import ClauseRecord, read_records
from legal_simplifier.prompts import build_completion, build_messages
from legal_simplifier.training.lora import (
    build_lora_config,
    build_quantization_config,
    check_lora_coverage,
    trainable_parameters,
)
from legal_simplifier.training.tracking import configure_tracking

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedModel
    from trl import SFTTrainer

LOGGER = logging.getLogger(__name__)

ADAPTER_DIR = "adapter"
MERGED_DIR = "merged"
FULL_MODEL_DIR = "model"


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


def _torch_dtype(name: str) -> Any:
    import torch

    return name if name == "auto" else getattr(torch, name)


def load_base_model(
    model: ModelConfig, quantization: QuantizationConfig | None = None
) -> PreTrainedModel:
    """The text-only causal LM, in 4 bit when ``quantization.load_in_4bit`` (QLoRA)."""

    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {"revision": model.revision, "dtype": _torch_dtype(model.dtype)}
    quant = build_quantization_config(quantization) if quantization else None
    if quant is not None:
        import torch

        kwargs["quantization_config"] = quant
        kwargs["device_map"] = {
            "": torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        }
    return AutoModelForCausalLM.from_pretrained(model.name_or_path, **kwargs)


def build_sft_trainer(config: SFTConfigFile, model: PreTrainedModel | None = None) -> SFTTrainer:
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

    if model is None:
        model = load_base_model(config.model, config.quantization)
    peft_config = None
    if config.lora.enabled:
        check_lora_coverage(model, config.lora.target_modules)
        peft_config = build_lora_config(config.lora, base_revision=config.model.revision)

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
        use_cpu=t.use_cpu,
        run_name=config.experiment.name,
        **configure_tracking(config.tracking),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name_or_path, revision=config.model.revision
    )
    return SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )


def merge_adapter(adapter_dir: Path | str, output_dir: Path | str, model: ModelConfig) -> Path:
    """Load the base in its training dtype (never 4 bit), apply the adapter, merge and save."""

    from peft import PeftModel
    from transformers import AutoTokenizer

    base = load_base_model(model)
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    output_dir = Path(output_dir)
    merged.save_pretrained(str(output_dir))
    AutoTokenizer.from_pretrained(str(adapter_dir)).save_pretrained(str(output_dir))
    LOGGER.info("Merged %s into %s", adapter_dir, output_dir)
    return output_dir


def train_sft(config: SFTConfigFile) -> Path:
    """Run SFT, save adapter / merged weights, metrics and model cards; return the run directory."""

    from legal_simplifier.training.model_card import write_model_cards

    model = load_base_model(config.model, config.quantization)
    lora_info: dict[str, Any] = {"enabled": config.lora.enabled}
    if config.lora.enabled:
        lora_info["coverage"] = check_lora_coverage(model, config.lora.target_modules).to_dict()

    trainer = build_sft_trainer(config, model=model)
    lora_info["parameters"] = trainable_parameters(trainer.model)  # type: ignore[arg-type]
    LOGGER.info("Starting SFT for %s (%s)", config.experiment.name, lora_info["parameters"])
    result = trainer.train()
    metrics: dict[str, Any] = dict(result.metrics)
    if trainer.eval_dataset is not None:
        metrics.update(trainer.evaluate())

    run_dir = Path(config.training.output_dir)
    weights_dir = run_dir / (ADAPTER_DIR if config.lora.enabled else FULL_MODEL_DIR)
    trainer.save_model(str(weights_dir))  # a PeftModel saves only the adapter
    trainer.processing_class.save_pretrained(str(weights_dir))  # type: ignore[union-attr]
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    (run_dir / "experiment.json").write_text(config.model_dump_json(indent=2))
    (run_dir / "lora.json").write_text(json.dumps(lora_info, indent=2))

    if config.lora.enabled and config.training.save_merged:
        del trainer, model  # free the training copy before loading the base again
        merge_adapter(weights_dir, run_dir / MERGED_DIR, config.model)
    write_model_cards(run_dir)
    LOGGER.info("Saved SFT run to %s", run_dir)
    return run_dir
