"""Fit a training config to the GPU it lands on (used by the notebook and the Modal script).

The real configs assume bf16 (L4, A10G, A100, H100). Colab's free T4 and
Kaggle's T4 / P100 have no bf16, so there the frozen base is kept in fp32 and
training runs with fp16 autocast. fp16 weights are avoided because Qwen
activations can overflow in fp16. A 0.8B model in fp32 with LoRA fits in 16 GB.
"""

from __future__ import annotations

import logging

from legal_simplifier.config import SFTConfigFile

LOGGER = logging.getLogger(__name__)


def bf16_available() -> bool:
    import torch

    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def adapt_precision(config: SFTConfigFile, *, bf16_supported: bool | None = None) -> SFTConfigFile:
    """Return a copy of ``config`` that runs on this GPU; unchanged unless bf16 is missing."""

    if bf16_supported is None:
        bf16_supported = bf16_available()
    wants_bf16 = (
        config.training.bf16
        or config.model.dtype == "bfloat16"
        or (config.quantization.load_in_4bit and config.quantization.compute_dtype == "bfloat16")
    )
    if bf16_supported or not wants_bf16:
        return config
    adapted = config.model_copy(deep=True)
    adapted.training.bf16 = False
    adapted.training.fp16 = True
    if adapted.model.dtype == "bfloat16":
        adapted.model.dtype = "float32"
    if adapted.quantization.compute_dtype == "bfloat16":
        adapted.quantization.compute_dtype = "float16"
    LOGGER.warning("This GPU has no bf16: training with fp16 autocast on fp32 weights instead.")
    return adapted
