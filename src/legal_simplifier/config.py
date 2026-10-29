"""Typed experiment configs, loaded from one YAML file per experiment.

Unknown keys are rejected so a typo in a YAML file fails fast instead of being
silently ignored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypeVar

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

    stage: Literal["sft"] = "sft"
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
    """Weights of the four reward functions; GRPO passes them as ``reward_weights``.

    A weight of 0 keeps the function computed and logged but out of the advantage,
    which is how the entity ablation stays comparable.
    """

    meaning: float = 1.0
    simplicity: float = 1.0
    entity_preservation: float = 1.0
    format: float = 1.0


class MeaningConfig(StrictModel):
    """How ``meaning`` is scored (see docs/decisions.md, ADR-018).

    * ``nli``: P(entailment | premise = clause, hypothesis = output). Catches added or
      contradicted content; blind to dropped content.
    * ``nli_coverage``: the clause split into sentences and conditional parts, each
      scored as P(entailment | premise = output, hypothesis = part), then averaged.
      Catches a dropped part; may also punish legitimate compression (the
      ``meaning_coverage`` ablation measures that).
    * ``bertscore``: F1 against the reference.
    """

    method: str = Field(default="nli", pattern="^(nli|nli_coverage|bertscore)$")
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    nli_revision: str | None = "6c749ce3425cd33b46d187e45b92bbf96ee12ec7"
    entailment_label: str | None = None
    """Label name of the entailment class; ``None`` finds the label containing "entail"."""
    max_length: int = 512
    batch_size: int = 32
    bertscore_model: str = "roberta-large"
    bertscore_rescale_with_baseline: bool = True
    coverage_max_parts: int = 8
    """``nli_coverage``: at most this many parts per clause (the rest is merged into the last)."""
    coverage_min_words: int = 3
    """``nli_coverage``: a shorter part is merged into its neighbour."""


class SimplicityConfig(StrictModel):
    """SARI against the reference, a reading-grade band, or their mean (``both``)."""

    method: str = Field(default="both", pattern="^(sari|fkgl_band|both)$")
    fkgl_low: float = 5.0
    fkgl_high: float = 9.0
    """Full score for an FKGL inside [low, high]; it falls linearly to 0 over ``fkgl_margin``."""
    fkgl_margin: float = 4.0

    @model_validator(mode="after")
    def _check_band(self) -> SimplicityConfig:
        if not self.fkgl_low < self.fkgl_high or self.fkgl_margin <= 0:
            raise ValueError("need fkgl_low < fkgl_high and fkgl_margin > 0")
        return self


class FormatConfig(StrictModel):
    """Penalties in [-1, 0]: empty output, verbatim copying, runaway length, and the
    reward-hacking detectors (see :mod:`legal_simplifier.rewards.hacking`)."""

    copy_threshold: float = 0.85
    """Token 4-gram overlap with the clause at or above which the output counts as a copy."""
    max_length_ratio: float = 1.6
    """Output words / clause words above which the output counts as runaway."""
    copy_penalty: float = 1.0
    runaway_penalty: float = 0.5
    truncation_penalty: float = 0.5
    word_salad_penalty: float = 1.0
    modality_flip_penalty: float = 0.5
    condition_drop_penalty: float = 0.5


class RewardConfig(StrictModel):
    """Reward functions shared by GRPO, DPO pair ranking and evaluation."""

    weights: RewardWeights = Field(default_factory=RewardWeights)
    meaning: MeaningConfig = Field(default_factory=MeaningConfig)
    simplicity: SimplicityConfig = Field(default_factory=SimplicityConfig)
    format: FormatConfig = Field(default_factory=FormatConfig)


class PolicyConfig(StrictModel):
    """GRPO and DPO keep training the SFT LoRA adapter on top of ``model`` (the base)."""

    sft_adapter: Path = Path("checkpoints/sft/adapter")


class RLDataConfig(StrictModel):
    train_path: Path = Path("data/processed/train.jsonl")
    include_category: bool = True
    max_train_samples: int | None = None
    """Use only the first N clauses (after a seeded shuffle); ``None`` uses all."""


class RLTrainingCommon(StrictModel):
    """Trainer settings shared by GRPO and DPO."""

    output_dir: Path
    learning_rate: float = 1.0e-5
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_train_epochs: float = 1.0
    max_steps: int = -1
    warmup_steps: float = 0.0
    lr_scheduler_type: str = "constant_with_warmup"
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    bf16: bool = False
    fp16: bool = False
    gradient_checkpointing: bool = False
    logging_steps: int = 5
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int | None = 2
    use_cpu: bool = False
    save_merged: bool = True


class GRPOTrainingConfig(RLTrainingCommon):
    output_dir: Path = Path("checkpoints/grpo")
    num_generations: int = 8
    max_completion_length: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    beta: float = 0.0
    """KL weight. 0 loads no reference; > 0 adds a frozen copy of the SFT adapter as reference."""
    loss_type: str = "dapo"
    epsilon: float = 0.2
    num_iterations: int = 1
    steps_per_generation: int | None = None
    scale_rewards: str = "group"
    multi_objective_aggregation: str = Field(
        default="sum_then_normalize", pattern="^(sum_then_normalize|normalize_then_sum)$"
    )
    mask_truncated_completions: bool = True
    """Completions that hit max_completion_length get no gradient (no reward for rambling)."""
    use_vllm: bool = False
    vllm_mode: str = Field(default="colocate", pattern="^(colocate|server)$")
    vllm_gpu_memory_utilization: float = 0.3
    log_completions: bool = True
    """Sample completions go to the tracker's table (never to the Hub)."""

    @model_validator(mode="after")
    def _check_group(self) -> GRPOTrainingConfig:
        per_generation = self.per_device_train_batch_size * (
            self.steps_per_generation or self.gradient_accumulation_steps
        )
        if self.num_generations < 2 or per_generation % self.num_generations:
            raise ValueError(
                f"num_generations ({self.num_generations}) must be >= 2 and divide the "
                f"completions per generation round ({per_generation} = batch x steps_per_generation)"
            )
        return self


class GRPOConfigFile(StrictModel):
    """Contents of ``configs/grpo.yaml`` (and the resolved ablation configs)."""

    stage: Literal["grpo"] = "grpo"
    experiment: ExperimentConfig
    data: RLDataConfig = Field(default_factory=RLDataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    rewards: RewardConfig = Field(default_factory=RewardConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    grpo: GRPOTrainingConfig = Field(default_factory=GRPOTrainingConfig)


class PairsConfig(StrictModel):
    """Preference pairs: sample N outputs per clause from the SFT model, rank by the reward."""

    output_dir: Path = Path("data/preferences")
    splits: list[str] = Field(default_factory=lambda: ["train", "dev"])
    data_dir: Path = Path("data/processed")
    max_clauses: int | None = None
    n_samples: int = 8
    temperature: float = 1.0
    top_p: float = 0.95
    max_new_tokens: int = 256
    batch_size: int = 8
    """Clauses per generate() call (each returns n_samples sequences)."""
    min_margin: float = 0.1
    """Skip a clause when best and worst composite rewards differ by less than this."""
    require_clean_chosen: bool = True
    """The chosen output must trip no format problem or reward-hacking detector."""

    @model_validator(mode="after")
    def _check(self) -> PairsConfig:
        if self.n_samples < 2:
            raise ValueError("n_samples must be at least 2 to form a pair")
        return self


class DPOTrainingConfig(RLTrainingCommon):
    output_dir: Path = Path("checkpoints/dpo")
    learning_rate: float = 5.0e-6
    per_device_train_batch_size: int = 4
    beta: float = 0.1
    loss_type: list[str] = Field(default_factory=lambda: ["sigmoid"])
    max_length: int = 1024
    eval_strategy: str = "steps"
    eval_steps: int = 50


class DPODataConfig(StrictModel):
    include_category: bool = True
    """The clauses come from ``pairs.data_dir``; this only shapes the prompt."""


class DPOConfigFile(StrictModel):
    """Contents of ``configs/dpo.yaml``: pair building and DPO training."""

    stage: Literal["dpo"] = "dpo"
    experiment: ExperimentConfig
    data: DPODataConfig = Field(default_factory=DPODataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    rewards: RewardConfig = Field(default_factory=RewardConfig)
    pairs: PairsConfig = Field(default_factory=PairsConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    dpo: DPOTrainingConfig = Field(default_factory=DPOTrainingConfig)


class SystemSpec(StrictModel):
    """One system under evaluation. Add a system by adding an entry (and, for a new kind,
    a class in :mod:`legal_simplifier.eval.systems`)."""

    name: str = Field(pattern=r"^[A-Za-z0-9_.+-]+$")
    kind: Literal["copy", "truncate", "hf", "claude"]
    group: str | None = None
    """Systems that differ only by training seed share a group (mean ± std over seeds)."""
    seed: int | None = None
    adapter: Path | None = None
    """``hf``: a LoRA adapter directory on the base model; ``None`` evaluates the base model."""
    few_shot: int = 0
    """``hf``: number of worked examples (from ``data.few_shot_path``) before the clause."""
    truncate_ratio: float = 0.5
    """``truncate``: keep this share of the clause's words."""
    trained: bool = False
    """Compared against ``reference_system`` with a paired test."""


class EvalDataConfig(StrictModel):
    golden_dir: Path = Path("data/golden")
    include_unreviewed: bool = True
    """Unreviewed golden items still have silver references; the manifest records the status."""
    dev_path: Path | None = None
    """Also evaluate on this split (e.g. data/processed/dev.jsonl); ``None`` = golden only."""
    dev_max_items: int | None = None
    few_shot_path: Path = Path("data/processed/train.jsonl")
    few_shot_seed: int = 7


class EvalGenerationConfig(StrictModel):
    """Greedy decoding for every local model, so outputs are reproducible and cacheable."""

    max_new_tokens: int = 384
    batch_size: int = 8


class EvalMetricsConfig(StrictModel):
    meaning: MeaningConfig = Field(default_factory=MeaningConfig)
    """NLI model for faithfulness (clause -> output) and coverage (each part <- output)."""
    bertscore_model: str | None = "roberta-large"
    """``None`` skips BERTScore."""
    bertscore_num_layers: int | None = None
    bertscore_rescale_with_baseline: bool = True
    bertscore_batch_size: int = 32
    faithfulness_threshold: float = 0.5
    """An output below this P(entailment) counts towards the hallucination rate."""
    n_bootstrap: int = 1000
    confidence: float = 0.95


class ClaudeSystemConfig(StrictModel):
    """The ``claude`` system: zero-shot through the Batches API, same prompt as the student."""

    model: str = "claude-sonnet-5-5"
    effort: str = Field(default="medium", pattern="^(low|medium|high|xhigh|max)$")
    max_tokens: int = 4096
    cache_dir: Path = Path("data/cache/anthropic")
    cost_log: Path = Path("data/cache/anthropic/cost_log.jsonl")
    poll_interval_s: float = 60.0


class JudgeConfig(StrictModel):
    """LLM judge. ``claude`` is the primary judge; ``hf`` is the open-weight cross-check;
    ``proxy`` is a deterministic stand-in for smoke runs and the CI gate (never a result)."""

    backend: Literal["claude", "hf", "proxy"] = "claude"
    model: str = "claude-opus-5-5"
    effort: str = Field(default="medium", pattern="^(low|medium|high|xhigh|max)$")
    max_tokens: int = 8192
    cache_dir: Path = Path("data/cache/anthropic")
    cost_log: Path = Path("data/cache/anthropic/cost_log.jsonl")
    poll_interval_s: float = 60.0
    hf_model: str = "ibm-granite/granite-3.3-8b-instruct"
    hf_revision: str | None = "51dd4bc2ade4059a6bd87649d68aa11e4fb2529b"
    hf_dtype: str = "bfloat16"
    hf_max_new_tokens: int = 1536
    hf_cache_dir: Path = Path("results/cache/judge_hf")
    seed: int = 11
    """Seeds the blind shuffle of outputs within each clause."""
    pilot_seed: int = 21
    low_score: int = 2
    """A criterion at or below this counts as a failure in the error analysis."""


class EvalConfigFile(StrictModel):
    """Contents of ``configs/eval.yaml``."""

    stage: Literal["eval"] = "eval"
    experiment: ExperimentConfig
    output_root: Path = Path("results")
    cache_dir: Path = Path("results/cache/generations")
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: EvalDataConfig = Field(default_factory=EvalDataConfig)
    systems: list[SystemSpec]
    reference_system: str = "sft"
    """Trained systems are tested against this one (paired bootstrap / permutation)."""
    generation: EvalGenerationConfig = Field(default_factory=EvalGenerationConfig)
    metrics: EvalMetricsConfig = Field(default_factory=EvalMetricsConfig)
    claude: ClaudeSystemConfig = Field(default_factory=ClaudeSystemConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)

    @model_validator(mode="after")
    def _check_systems(self) -> EvalConfigFile:
        names = [s.name for s in self.systems]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate system names: {names}")
        if any(s.trained for s in self.systems) and self.reference_system not in names:
            raise ValueError(f"reference_system {self.reference_system!r} is not a system")
        return self


class OTelConfig(StrictModel):
    """Optional OpenTelemetry tracing (the ``otel`` extra). Off unless enabled here."""

    enabled: bool = False
    service_name: str = "legal-clause-simplifier"
    endpoint: str | None = None
    """OTLP/HTTP endpoint, e.g. http://localhost:4318; ``None`` uses the OTEL_* environment."""


class ServeConfig(StrictModel):
    """Contents of ``configs/serve.yaml``: the model behind ``lcs serve`` and the Space."""

    stage: Literal["serve"] = "serve"
    model: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            name_or_path="checkpoints/grpo/merged", dtype="bfloat16"
        )
    )
    model_label: str = "sft-grpo"
    """The system name reported by the API and used to match benchmark cost to eval results."""
    engine: Literal["auto", "transformers", "vllm"] = "auto"
    """``auto``: vLLM when a CUDA GPU and vllm are present (falling back if it cannot load the
    model), otherwise transformers."""
    max_new_tokens: int = 384
    max_clause_chars: int = 4000
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    vllm_gpu_memory_utilization: float = 0.85
    vllm_max_model_len: int = 4096
    otel: OTelConfig = Field(default_factory=OTelConfig)


class BenchmarkConfig(StrictModel):
    """Contents of ``configs/benchmark.yaml``: latency, throughput and cost of the served model."""

    stage: Literal["benchmark"] = "benchmark"
    experiment: ExperimentConfig
    serve: ServeConfig = Field(default_factory=ServeConfig)
    clauses_path: Path = Path("data/processed/test.jsonl")
    n_clauses: int = 200
    warmup: int = 5
    batch_size: int = 16
    """Clauses per call in the throughput pass (the latency pass sends one at a time)."""
    gpu: str = "NVIDIA L4"
    gpu_hourly_usd: float | None = None
    """What the stated GPU costs per hour where it runs. Required for a cost figure; the default
    config cites its source. ``None`` reports cost as unknown."""
    gpu_price_source: str | None = None
    eval_run: Path | None = None
    """An eval run with a ``claude`` system: its real cost from the cost log is the comparison."""
    cost_log: Path = Path("data/cache/anthropic/cost_log.jsonl")
    claude_model: str = "claude-sonnet-5-5"
    output_dir: Path = Path("results/benchmarks")


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
