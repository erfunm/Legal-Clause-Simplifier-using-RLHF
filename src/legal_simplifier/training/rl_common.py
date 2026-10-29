"""Shared pieces of the GRPO and DPO stages: the policy, prompt data, memory logging, saving.

Both stages keep training the **SFT adapter**: the base model is loaded and the
SFT LoRA weights are attached as the trainable ``default`` adapter. With
``beta > 0`` TRL adds a frozen copy of that adapter (``ref``) and uses it as the
reference, so the KL term pulls towards the SFT policy (not the raw base) and
no second copy of the 0.8B weights is loaded. Outputs mirror SFT: ``adapter/``,
``merged/``, model cards, ``metrics.json``, ``experiment.json`` and
``log_history.json`` (every logged step, including ``rewards/<name>/mean``).
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from legal_simplifier.config import ModelConfig, RLDataConfig
from legal_simplifier.data.schema import ClauseRecord, read_records
from legal_simplifier.prompts import build_messages
from legal_simplifier.training.sft import ADAPTER_DIR, MERGED_DIR, load_base_model, merge_adapter

if TYPE_CHECKING:
    from peft import PeftModel
    from pydantic import BaseModel
    from transformers import Trainer, TrainerCallback

LOGGER = logging.getLogger(__name__)


def check_adapter_base(adapter_dir: Path | str, model: ModelConfig) -> dict[str, Any]:
    """Refuse an adapter trained on a different base model or revision than ``model``."""

    path = Path(adapter_dir) / "adapter_config.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: train SFT first (lcs train-sft)")
    adapter = json.loads(path.read_text(encoding="utf-8"))
    base, revision = adapter.get("base_model_name_or_path"), adapter.get("revision")
    if base != model.name_or_path or (model.revision and revision and revision != model.revision):
        raise ValueError(
            f"{adapter_dir} was trained on {base}@{revision}, but the config's base is "
            f"{model.name_or_path}@{model.revision}"
        )
    return adapter


def load_policy(model: ModelConfig, sft_adapter: Path | str) -> PeftModel:
    """The base model with the SFT adapter attached as the trainable ``default`` adapter."""

    from peft import PeftModel

    check_adapter_base(sft_adapter, model)
    base = load_base_model(model)
    return PeftModel.from_pretrained(base, str(sft_adapter), is_trainable=True)


def load_sft_for_sampling(model: ModelConfig, sft_adapter: Path | str) -> Any:
    """The SFT model with its adapter merged in memory (faster to sample from), in eval mode."""

    from peft import PeftModel

    check_adapter_base(sft_adapter, model)
    merged = PeftModel.from_pretrained(load_base_model(model), str(sft_adapter)).merge_and_unload()
    return merged.eval()


def select_records(
    data: RLDataConfig, seed: int, path: Path | str | None = None
) -> list[ClauseRecord]:
    records = read_records(path or data.train_path)
    if not records:
        raise ValueError(f"no records in {path or data.train_path}")
    if data.max_train_samples is not None:
        records = list(records)
        random.Random(seed).shuffle(records)
        records = records[: data.max_train_samples]
    return records


def prompt_rows(
    records: Sequence[ClauseRecord],
    *,
    include_category: bool = True,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Conversational prompts plus the columns the reward functions read."""

    rows = []
    for record in records:
        row: dict[str, Any] = {
            "prompt": build_messages(record.clause, record.category if include_category else None),
            "clause": record.clause,
            "reference": record.reference,
            "entities": list(record.entities),
            "category": record.category,
            "id": record.id,
        }
        if chat_template_kwargs:
            row["chat_template_kwargs"] = chat_template_kwargs
        rows.append(row)
    return rows


def memory_callback() -> TrainerCallback:
    """Adds ``memory/max_allocated_gb`` to every log on CUDA (ADR-002: record the real peak)."""

    import torch
    from transformers import TrainerCallback

    class PeakMemory(TrainerCallback):
        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> None:
            if logs is not None and torch.cuda.is_available():
                logs["memory/max_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)

    return PeakMemory()


def save_rl_run(
    trainer: Trainer,
    config: BaseModel,
    *,
    model: ModelConfig,
    run_dir: Path | str,
    metrics: dict[str, Any],
    save_merged: bool,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write adapter/, metrics, experiment, log history, merged/ and the model cards."""

    from legal_simplifier.training.model_card import write_model_cards

    run_dir = Path(run_dir)
    adapter_dir = run_dir / ADAPTER_DIR
    peft_model = trainer.model
    if "ref" in getattr(peft_model, "peft_config", {}):  # TRL's frozen reference copy (beta > 0)
        peft_model.set_adapter("default")  # type: ignore[union-attr, operator]
        peft_model.delete_adapter("ref")  # type: ignore[union-attr, operator]
    trainer.save_model(str(adapter_dir))  # a PeftModel saves only its trainable adapter
    trainer.processing_class.save_pretrained(str(adapter_dir))  # type: ignore[union-attr]
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    (run_dir / "experiment.json").write_text(config.model_dump_json(indent=2))
    (run_dir / "log_history.json").write_text(
        json.dumps(trainer.state.log_history, indent=2, default=float)
    )
    if extra:
        (run_dir / "run_info.json").write_text(json.dumps(extra, indent=2, default=str))
    if save_merged:
        merge_adapter(adapter_dir, run_dir / MERGED_DIR, model)
    write_model_cards(run_dir)
    LOGGER.info("Saved run to %s", run_dir)
    return run_dir
