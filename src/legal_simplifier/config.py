"""Typed experiment configs, loaded from one YAML file per experiment.

Unknown keys are rejected so a typo in a YAML file fails fast instead of being
silently ignored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

T = TypeVar("T", bound=BaseModel)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExperimentConfig(StrictModel):
    name: str
    seed: int = 42


class DataConfig(StrictModel):
    train_path: Path = Path("data/processed/train.jsonl")
    dev_path: Path = Path("data/processed/dev.jsonl")
    include_category: bool = True
    """Mention the clause category in the prompt."""


class ModelConfig(StrictModel):
    name_or_path: str = "Qwen/Qwen3.5-0.8B"
    revision: str | None = None
    dtype: str = "auto"
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Extra arguments to ``apply_chat_template``, e.g. ``{"enable_thinking": false}`` for Qwen3."""


class SFTTrainingConfig(StrictModel):
    output_dir: Path = Path("checkpoints/sft")
    max_length: int = 1024
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_train_epochs: float = 3.0
    max_steps: int = -1
    learning_rate: float = 2.0e-4
    weight_decay: float = 0.0
    warmup_steps: float = 0.03
    """An int is a step count; a float in [0, 1) is a fraction of total steps."""
    lr_scheduler_type: str = "cosine"
    bf16: bool = False
    fp16: bool = False
    gradient_checkpointing: bool = False
    logging_steps: int = 10
    eval_strategy: str = "epoch"
    eval_steps: int | None = None
    save_strategy: str = "epoch"
    save_total_limit: int | None = 2
    report_to: list[str] = Field(default_factory=lambda: ["none"])
    use_cpu: bool = False


class SFTConfigFile(StrictModel):
    """Contents of ``configs/sft*.yaml``."""

    experiment: ExperimentConfig
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: SFTTrainingConfig = Field(default_factory=SFTTrainingConfig)


class RewardWeights(StrictModel):
    bertscore: float = 0.5
    sari: float = 0.3
    readability: float = 0.2
    entity_penalty: float = 1.0


class RewardConfig(StrictModel):
    """Composite reward settings shared by GRPO, DPO pair ranking and evaluation."""

    weights: RewardWeights = Field(default_factory=RewardWeights)
    bertscore_model: str = "roberta-large"
    bertscore_rescale_with_baseline: bool = True
    bertscore_batch_size: int = 32
    readability_target_grade: float = 8.0
    missing_entity_penalty: float = 1.0
    """Penalty when every entity is missing; scaled by the fraction missing."""
    clip_min: float = -1.0
    clip_max: float = 1.5

    @model_validator(mode="after")
    def _check_clip(self) -> RewardConfig:
        if self.clip_min >= self.clip_max:
            raise ValueError("clip_min must be below clip_max")
        return self


def load_yaml(path: Path | str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def load_config(path: Path | str, model: type[T]) -> T:
    """Load ``path`` and validate it against ``model``."""

    return model.model_validate(load_yaml(path))
