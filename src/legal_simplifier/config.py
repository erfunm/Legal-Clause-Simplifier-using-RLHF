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
    smoke: bool = False
    """A CPU smoke run with a tiny random model. Its outputs are never results."""


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
    save_merged: bool = True
    """With LoRA, also write base + adapter merged into plain weights under ``merged/``."""


# Every projection in Qwen3.5's two layer types: full attention (q/k/v/o), Gated DeltaNet
# linear attention (in_proj_qkv, in_proj_z, out_proj) and the MLP. Left out on purpose:
# in_proj_a / in_proj_b (per-head gate scalars, a few outputs wide), conv1d, and the
# tied embeddings / lm_head. See docs/decisions.md, ADR-016.
QWEN35_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


class LoRAConfig(StrictModel):
    """PEFT LoRA adapter. ``enabled: false`` trains every weight instead."""

    enabled: bool = True
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = Field(default_factory=lambda: list(QWEN35_LORA_TARGETS))
    """Module-name suffixes. Each one must match at least one linear layer."""
    use_rslora: bool = False


class QuantizationConfig(StrictModel):
    """QLoRA: load the frozen base in 4 bit with bitsandbytes (needs a CUDA GPU)."""

    load_in_4bit: bool = False
    quant_type: str = Field(default="nf4", pattern="^(nf4|fp4)$")
    double_quant: bool = True
    compute_dtype: str = Field(default="bfloat16", pattern="^(bfloat16|float16|float32)$")


class TrackingConfig(StrictModel):
    """Experiment tracking. Keys come from the environment (``WANDB_API_KEY``), never config."""

    backend: str = Field(default="none", pattern="^(none|wandb|trackio)$")
    project: str = "legal-clause-simplifier"
    wandb_entity: str | None = None
    wandb_mode: str = Field(default="online", pattern="^(online|offline)$")
    trackio_dir: Path | None = None
    """Local Trackio store; ``None`` keeps Trackio's default."""
    trackio_space_id: str | None = None
    """Setting this publishes the dashboard as a Hugging Face Space: ask first."""


class SFTConfigFile(StrictModel):
    """Contents of ``configs/sft*.yaml``."""

    experiment: ExperimentConfig
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    training: SFTTrainingConfig = Field(default_factory=SFTTrainingConfig)

    @model_validator(mode="after")
    def _qlora_needs_lora(self) -> SFTConfigFile:
        if self.quantization.load_in_4bit and not self.lora.enabled:
            raise ValueError("quantization.load_in_4bit (QLoRA) requires lora.enabled")
        return self


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


class ExtractionConfig(StrictModel):
    cuad_json: Path | None = None
    """Local ``CUAD_v1.json``; ``None`` downloads the pinned file from the Hub."""
    categories: list[str] | None = None
    """``None`` uses ``legal_simplifier.data.cuad.DEFAULT_CATEGORIES``."""
    min_chars: int = 40
    min_words: int = 6
    max_chars: int = 1500


class SplitConfig(StrictModel):
    dedup_threshold: float = Field(default=0.8, gt=0.0, le=1.0)
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    seed: int = 13


class ReferenceConfig(StrictModel):
    """Silver reference generation."""

    backend: str = Field(default="mock", pattern="^(mock|claude)$")
    model: str = "claude-sonnet-5-5"
    effort: str = Field(default="medium", pattern="^(low|medium|high|xhigh|max)$")
    max_tokens: int = 4096
    """Includes adaptive thinking tokens, so leave headroom over the clause length."""
    cache_dir: Path = Path("data/cache/anthropic")
    cost_log: Path = Path("data/cache/anthropic/cost_log.jsonl")
    poll_interval_s: float = 60.0
    pilot_seed: int = 21
    """Seed for the category-balanced ``--limit`` pilot sample."""


class GoldenConfig(StrictModel):
    output_dir: Path = Path("data/golden")
    size: int = Field(default=150, gt=0)
    seed: int = 7


class DataPrepConfig(StrictModel):
    """Contents of ``configs/data*.yaml``."""

    output_dir: Path = Path("data/processed")
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    references: ReferenceConfig = Field(default_factory=ReferenceConfig)
    golden: GoldenConfig = Field(default_factory=GoldenConfig)


def load_yaml(path: Path | str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def load_config(path: Path | str, model: type[T]) -> T:
    """Load ``path`` and validate it against ``model``."""

    return model.model_validate(load_yaml(path))
