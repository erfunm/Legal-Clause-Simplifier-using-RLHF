"""LoRA / QLoRA setup with PEFT, and a check of which layers the adapter actually wraps.

``target_modules`` are name suffixes, so a typo or a model with different
layer names would silently train fewer layers than intended. ``lora_coverage``
lists every linear layer of the base model as wrapped or left out and raises
if a configured target matches nothing.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from legal_simplifier.config import LoRAConfig, QuantizationConfig

if TYPE_CHECKING:
    import torch
    from peft import LoraConfig as PeftLoraConfig
    from transformers import BitsAndBytesConfig


def build_lora_config(config: LoRAConfig, *, base_revision: str | None = None) -> PeftLoraConfig:
    """PEFT config; ``base_revision`` is saved in ``adapter_config.json`` to pin the base."""

    from peft import LoraConfig as PeftLoraConfig

    return PeftLoraConfig(
        task_type="CAUSAL_LM",
        revision=base_revision,
        r=config.r,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=list(config.target_modules),
        use_rslora=config.use_rslora,
        bias="none",
    )


def build_quantization_config(config: QuantizationConfig) -> BitsAndBytesConfig | None:
    """The 4-bit bitsandbytes config for QLoRA, or ``None`` for a bf16/fp32 base."""

    if not config.load_in_4bit:
        return None
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.quant_type,
        bnb_4bit_use_double_quant=config.double_quant,
        bnb_4bit_compute_dtype=getattr(torch, config.compute_dtype),
    )


@dataclass
class LoRACoverage:
    """Which linear layers of the base model a LoRA config wraps."""

    wrapped: dict[str, int] = field(default_factory=dict)
    """Layer-name pattern (digits replaced by ``*``) -> number of layers wrapped."""
    left_out: dict[str, int] = field(default_factory=dict)
    unmatched_targets: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pattern(name: str) -> str:
    return ".".join("*" if part.isdigit() else part for part in name.split("."))


def lora_coverage(model: torch.nn.Module, target_modules: Sequence[str]) -> LoRACoverage:
    """Match ``target_modules`` (name suffixes, as PEFT does) against the model's linear layers."""

    import torch

    wrapped: Counter[str] = Counter()
    left_out: Counter[str] = Counter()
    hits: Counter[str] = Counter()
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf in target_modules:
            wrapped[_pattern(name)] += 1
            hits[leaf] += 1
        else:
            left_out[_pattern(name)] += 1
    return LoRACoverage(
        wrapped=dict(sorted(wrapped.items())),
        left_out=dict(sorted(left_out.items())),
        unmatched_targets=[t for t in target_modules if not hits[t]],
    )


def check_lora_coverage(model: torch.nn.Module, target_modules: Sequence[str]) -> LoRACoverage:
    coverage = lora_coverage(model, target_modules)
    if coverage.unmatched_targets:
        raise ValueError(
            f"LoRA target_modules {coverage.unmatched_targets} match no linear layer of "
            f"{type(model).__name__}; check the names for this architecture"
        )
    return coverage


def trainable_parameters(model: torch.nn.Module) -> dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total}
