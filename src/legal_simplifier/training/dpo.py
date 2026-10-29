"""DPO from the SFT adapter on reward-ranked preference pairs (``lcs build-preferences``).

The SFT adapter is the trainable policy; TRL keeps a frozen copy of it as the
reference (``ref`` adapter), so the implicit KL is to the SFT model, matching
how the pairs were sampled. TRL's DPO loss uses the same fused, chunked LM
head as GRPO and SFT, so the 248k-vocabulary logits are never materialised
(ADR-002).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from legal_simplifier.config import DPOConfigFile
from legal_simplifier.training.rl_common import load_policy, memory_callback, save_rl_run
from legal_simplifier.training.tracking import configure_tracking

if TYPE_CHECKING:
    from datasets import Dataset
    from trl import DPOTrainer

LOGGER = logging.getLogger(__name__)

PAIR_COLUMNS = ("prompt", "chosen", "rejected")


def load_pairs(path: Path, chat_template_kwargs: dict[str, Any] | None = None) -> Dataset | None:
    """The pair file as a dataset with only the columns DPO reads; ``None`` if absent or empty."""

    import json

    from datasets import Dataset

    if not path.exists():
        return None
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                kept = {k: row[k] for k in PAIR_COLUMNS}
                if chat_template_kwargs:
                    kept["chat_template_kwargs"] = chat_template_kwargs
                rows.append(kept)
    return Dataset.from_list(rows) if rows else None


def build_dpo_trainer(config: DPOConfigFile, *, model: Any | None = None) -> DPOTrainer:
    from transformers import AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    kwargs = config.model.chat_template_kwargs
    train = load_pairs(config.pairs.output_dir / "train.jsonl", kwargs)
    if train is None:
        raise ValueError(
            f"no preference pairs in {config.pairs.output_dir / 'train.jsonl'}: "
            "run `lcs build-preferences` first (or lower pairs.min_margin)"
        )
    dev = load_pairs(config.pairs.output_dir / "dev.jsonl", kwargs)
    policy = model if model is not None else load_policy(config.model, config.policy.sft_adapter)

    d = config.dpo
    args = DPOConfig(
        output_dir=str(d.output_dir),
        seed=config.experiment.seed,
        run_name=config.experiment.name,
        learning_rate=d.learning_rate,
        per_device_train_batch_size=d.per_device_train_batch_size,
        per_device_eval_batch_size=d.per_device_train_batch_size,
        gradient_accumulation_steps=d.gradient_accumulation_steps,
        num_train_epochs=d.num_train_epochs,
        max_steps=d.max_steps,
        warmup_steps=d.warmup_steps,
        lr_scheduler_type=d.lr_scheduler_type,
        weight_decay=d.weight_decay,
        max_grad_norm=d.max_grad_norm,
        bf16=d.bf16,
        fp16=d.fp16,
        gradient_checkpointing=d.gradient_checkpointing,
        logging_steps=d.logging_steps,
        save_strategy=d.save_strategy,
        save_steps=d.save_steps,
        save_total_limit=d.save_total_limit,
        eval_strategy=d.eval_strategy if dev is not None else "no",
        eval_steps=d.eval_steps,
        use_cpu=d.use_cpu,
        beta=d.beta,
        loss_type=list(d.loss_type),
        max_length=d.max_length,
        **configure_tracking(config.tracking),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name_or_path, revision=config.model.revision
    )
    return DPOTrainer(
        model=policy,
        args=args,
        train_dataset=train,
        eval_dataset=dev,
        processing_class=tokenizer,
        callbacks=[memory_callback()],
    )


def train_dpo(config: DPOConfigFile) -> Path:
    trainer = build_dpo_trainer(config)
    LOGGER.info("Starting DPO %s: beta=%s", config.experiment.name, config.dpo.beta)
    result = trainer.train()
    metrics = dict(result.metrics)
    if trainer.eval_dataset is not None:
        metrics.update(trainer.evaluate())
    return save_rl_run(
        trainer,
        config,
        model=config.model,
        run_dir=config.dpo.output_dir,
        metrics=metrics,
        save_merged=config.dpo.save_merged,
        extra={
            "sft_adapter": str(config.policy.sft_adapter),
            "pairs_dir": str(config.pairs.output_dir),
        },
    )
