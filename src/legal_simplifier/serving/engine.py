"""Inference engines behind ``lcs serve``, the Space and ``lcs benchmark``.

* :class:`TransformersEngine`: any machine; greedy decoding, real token streaming
  through ``TextIteratorStreamer``.
* :class:`VLLMEngine`: CUDA + the ``vllm`` package (GPU image only). Batched
  generation; ``stream`` yields the finished text as one chunk.

:func:`make_engine` with ``engine: auto`` picks vLLM when a CUDA GPU and vllm are
present, and falls back to transformers (with a warning) if vLLM cannot load
the model; it is not yet verified that vLLM supports the text-only Qwen3.5
checkpoint SFT writes (docs/decisions.md, ADR-019 and ADR-028).

Every engine renders prompts with :func:`legal_simplifier.prompts.build_messages`,
the same prompt as training and evaluation, and reports token counts.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from legal_simplifier.config import ServeConfig
from legal_simplifier.prompts import Message
from legal_simplifier.rewards.suite import completion_text

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Generation:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str  # "stop" | "length"


class Engine(Protocol):
    name: str
    model_id: str

    def generate(
        self, batch: Sequence[Sequence[Message]], max_new_tokens: int
    ) -> list[Generation]: ...

    def stream(
        self, messages: Sequence[Message], max_new_tokens: int
    ) -> Iterator[str | Generation]:
        """Text deltas, then the final :class:`Generation`."""
        ...


class TransformersEngine:
    name = "transformers"

    def __init__(
        self, config: ServeConfig, *, model: Any | None = None, tokenizer: Any | None = None
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        m = config.model
        self.model_id = m.name_or_path
        self.chat_template_kwargs = m.chat_template_kwargs
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            m.name_or_path, revision=m.revision
        )
        if model is None:
            dtype = m.dtype if m.dtype == "auto" else getattr(torch, m.dtype)
            if not torch.cuda.is_available() and dtype in (torch.bfloat16, torch.float16):
                dtype = torch.float32  # half precision on CPU is slow and poorly supported
            model = AutoModelForCausalLM.from_pretrained(
                m.name_or_path, revision=m.revision, dtype=dtype
            )
            if torch.cuda.is_available():
                model = model.to("cuda")  # pyright: ignore[reportArgumentType]
        self.model = model.eval()  # pyright: ignore[reportOptionalMemberAccess]
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.lock = threading.Lock()  # one generate() at a time on one model

    def _encode(self, batch: Sequence[Sequence[Message]]) -> Any:
        texts = [
            self.tokenizer.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=True,
                **self.chat_template_kwargs,
            )
            for messages in batch
        ]
        device = next(self.model.parameters()).device
        return self.tokenizer(
            texts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(device)

    def generate(self, batch: Sequence[Sequence[Message]], max_new_tokens: int) -> list[Generation]:
        import torch

        enc = self._encode(batch)
        with self.lock, torch.inference_mode():
            out = self.model.generate(
                **enc,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        width = enc["input_ids"].shape[1]
        results = []
        for row in range(len(batch)):
            new = out[row, width:]
            n_new = int((new != self.tokenizer.pad_token_id).sum())
            text = self.tokenizer.decode(new, skip_special_tokens=True)
            results.append(
                Generation(
                    completion_text(text),
                    int(enc["attention_mask"][row].sum()),
                    n_new,
                    "length" if n_new >= max_new_tokens else "stop",
                )
            )
        return results

    def stream(
        self, messages: Sequence[Message], max_new_tokens: int
    ) -> Iterator[str | Generation]:
        import torch
        from transformers import TextIteratorStreamer

        enc = self._encode([messages])
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        output: dict[str, Any] = {}

        def run() -> None:
            with self.lock, torch.inference_mode():
                output["ids"] = self.model.generate(
                    **enc,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    streamer=streamer,
                )

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        pieces = []
        for delta in streamer:
            if delta:
                pieces.append(delta)
                yield delta
        thread.join()
        n_new = int(output["ids"].shape[1] - enc["input_ids"].shape[1])
        yield Generation(
            completion_text("".join(pieces)),
            int(enc["attention_mask"][0].sum()),
            n_new,
            "length" if n_new >= max_new_tokens else "stop",
        )


class VLLMEngine:
    """vLLM offline engine on one GPU (the GPU image installs vllm)."""

    name = "vllm"

    def __init__(self, config: ServeConfig):
        from vllm import LLM  # pyright: ignore[reportMissingImports]

        m = config.model
        self.model_id = m.name_or_path
        self.chat_template_kwargs = m.chat_template_kwargs
        self.llm = LLM(
            model=m.name_or_path,
            revision=m.revision,
            dtype=m.dtype,
            gpu_memory_utilization=config.vllm_gpu_memory_utilization,
            max_model_len=config.vllm_max_model_len,
        )

    def generate(self, batch: Sequence[Sequence[Message]], max_new_tokens: int) -> list[Generation]:
        from vllm import SamplingParams  # pyright: ignore[reportMissingImports]

        params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        outputs = self.llm.chat(
            [list(m) for m in batch], params, chat_template_kwargs=self.chat_template_kwargs or None
        )
        results = []
        for out in outputs:
            choice = out.outputs[0]
            results.append(
                Generation(
                    completion_text(choice.text),
                    len(out.prompt_token_ids or []),
                    len(choice.token_ids),
                    "length" if choice.finish_reason == "length" else "stop",
                )
            )
        return results

    def stream(
        self, messages: Sequence[Message], max_new_tokens: int
    ) -> Iterator[str | Generation]:
        (result,) = self.generate([messages], max_new_tokens)
        yield result.text
        yield result


def vllm_available() -> bool:
    import importlib.util

    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available() and importlib.util.find_spec("vllm") is not None


def make_engine(config: ServeConfig) -> Engine:
    if config.engine == "vllm" or (config.engine == "auto" and vllm_available()):
        try:
            return VLLMEngine(config)
        except Exception as exc:
            if config.engine == "vllm":
                raise
            LOGGER.warning(
                "vLLM could not load %s (%s); using transformers", config.model.name_or_path, exc
            )
    return TransformersEngine(config)
