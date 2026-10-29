"""Fit a training config to the GPU it lands on (used by the notebooks and the Modal scripts).

The real configs assume bf16 (L4, A10G, A100, H100). Colab's free T4 and
Kaggle's T4 / P100 have no bf16, so there the frozen base is kept in fp32 and
training runs with fp16 autocast. fp16 weights are avoided because Qwen
activations can overflow in fp16. A 0.8B model in fp32 with LoRA fits in 16 GB.
Works for every stage: the trainer section is ``training`` (SFT), ``grpo`` or ``dpo``.
"""

from __future__ import annotations

import logging
from typing import TypeVar

from legal_simplifier.config import DPOConfigFile, GRPOConfigFile, SFTConfigFile

LOGGER = logging.getLogger(__name__)

C = TypeVar("C", SFTConfigFile, GRPOConfigFile, DPOConfigFile)
TRAINER_SECTION = {"sft": "training", "grpo": "grpo", "dpo": "dpo"}


def bf16_available() -> bool:
    import torch

    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def adapt_precision(config: C, *, bf16_supported: bool | None = None) -> C:
    """Return a copy of ``config`` that runs on this GPU; unchanged unless bf16 is missing."""

    if bf16_supported is None:
        bf16_supported = bf16_available()
    trainer = getattr(config, TRAINER_SECTION[config.stage])
    quantization = getattr(config, "quantization", None)
    wants_bf16 = (
        trainer.bf16
        or config.model.dtype == "bfloat16"
        or (
            quantization is not None
            and quantization.load_in_4bit
            and quantization.compute_dtype == "bfloat16"
        )
    )
    if bf16_supported or not wants_bf16:
        return config
    adapted = config.model_copy(deep=True)
    section = getattr(adapted, TRAINER_SECTION[adapted.stage])
    section.bf16 = False
    section.fp16 = True
    if adapted.model.dtype == "bfloat16":
        adapted.model.dtype = "float32"
    adapted_quant = getattr(adapted, "quantization", None)
    if adapted_quant is not None and adapted_quant.compute_dtype == "bfloat16":
        adapted_quant.compute_dtype = "float16"
    LOGGER.warning("This GPU has no bf16: training with fp16 autocast on fp32 weights instead.")
    return adapted
