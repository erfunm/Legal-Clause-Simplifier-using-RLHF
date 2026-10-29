"""Systems under evaluation, each a small class, and the generation cache.

* ``copy``: the clause itself (lower bound: perfect entities, no simplification).
* ``truncate``: the first ``truncate_ratio`` of the clause's words (lower bound:
  shorter and "simpler", but drops content).
* ``hf``: the base model, optionally with a LoRA adapter (SFT, SFT+DPO, SFT+GRPO)
  and optionally few-shot, greedy decoding, same prompt as training.
* ``claude``: Claude zero-shot through the Batches API with the same prompt as the
  student (the upper reference). It is paid, so the CLI gates it.

Every system has a ``fingerprint``: a hash of everything that changes its outputs
(model id and revision, a hash of the adapter weights, the prompt, few-shot
examples, decoding settings). Outputs are cached under
``<cache_dir>/<system>/<fingerprint>.jsonl`` keyed by the item, so re-scoring,
re-judging or re-reporting never re-generates, and changing the adapter does.

To add a system: subclass :class:`System`, implement ``_identity`` and
``generate``, and register its ``kind`` in :func:`make_system`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from legal_simplifier.config import EvalConfigFile, SystemSpec
from legal_simplifier.eval.items import EvalItem
from legal_simplifier.prompts import SYSTEM_PROMPT, Message, build_completion, build_messages
from legal_simplifier.rewards.suite import completion_text

LOGGER = logging.getLogger(__name__)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class System:
    """Base class. Subclasses say what identifies them and how they generate."""

    paid = False

    def __init__(self, spec: SystemSpec):
        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec.name

    def _identity(self) -> dict[str, Any]:
        raise NotImplementedError

    @property
    def fingerprint(self) -> str:
        identity = {"kind": self.spec.kind, "prompt": SYSTEM_PROMPT, **self._identity()}
        canonical = json.dumps(identity, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.spec.kind,
            "group": self.spec.group,
            "seed": self.spec.seed,
            "trained": self.spec.trained,
            "fingerprint": self.fingerprint,
            **{k: v for k, v in self._identity().items() if k != "few_shot_examples"},
        }

    def generate(self, items: Sequence[EvalItem]) -> Sequence[str | None]:
        """One output per item; ``None`` marks a failure (scored as empty, never cached)."""

        raise NotImplementedError


class CopySystem(System):
    def _identity(self) -> dict[str, Any]:
        return {}

    def generate(self, items: Sequence[EvalItem]) -> Sequence[str | None]:
        return [" ".join(i.clause.split()) for i in items]


class TruncateSystem(System):
    def _identity(self) -> dict[str, Any]:
        return {"ratio": self.spec.truncate_ratio}

    def generate(self, items: Sequence[EvalItem]) -> Sequence[str | None]:
        out = []
        for item in items:
            words = item.clause.split()
            out.append(" ".join(words[: max(1, round(len(words) * self.spec.truncate_ratio))]))
        return out


def few_shot_examples(path: Path, k: int, seed: int) -> list[dict[str, str]]:
    """``k`` worked examples from the training split, one per category where possible."""

    from legal_simplifier.data.sampling import balanced_sample
    from legal_simplifier.data.schema import read_records

    if k <= 0:
        return []
    records = balanced_sample(read_records(path), k, seed=seed)
    random.Random(seed).shuffle(records)
    return [{"clause": r.clause, "category": r.category, "reference": r.reference} for r in records]


def build_prompt(item: EvalItem, examples: Sequence[dict[str, str]] = ()) -> list[Message]:
    """The training prompt, with optional worked examples as earlier chat turns."""

    messages = build_messages(item.clause, item.category)
    shots: list[Message] = []
    for ex in examples:
        shots.append(build_messages(ex["clause"], ex["category"])[1])
        shots.extend(build_completion(ex["reference"]))
    return [messages[0], *shots, messages[1]]


class HFSystem(System):
    """The base model with an optional LoRA adapter, greedy decoding."""

    def __init__(self, spec: SystemSpec, config: EvalConfigFile):
        super().__init__(spec)
        self.config = config
        self.examples = few_shot_examples(
            config.data.few_shot_path, spec.few_shot, config.data.few_shot_seed
        )

    def _identity(self) -> dict[str, Any]:
        adapter = self.spec.adapter
        weights = adapter / "adapter_model.safetensors" if adapter else None
        return {
            "model": self.config.model.name_or_path,
            "revision": self.config.model.revision,
            "dtype": self.config.model.dtype,
            "adapter": str(adapter) if adapter else None,
            "adapter_sha256": file_sha256(weights) if weights and weights.exists() else None,
            "few_shot": self.spec.few_shot,
            "few_shot_examples": [e["clause"][:80] for e in self.examples],
            "generation": self.config.generation.model_dump(),
            "chat_template_kwargs": self.config.model.chat_template_kwargs,
        }

    def _load(self) -> tuple[Any, Any]:
        from transformers import AutoTokenizer

        from legal_simplifier.training.sft import load_base_model

        model_cfg = self.config.model
        tokenizer = AutoTokenizer.from_pretrained(
            model_cfg.name_or_path, revision=model_cfg.revision
        )
        model = load_base_model(model_cfg)
        if self.spec.adapter is not None:
            from peft import PeftModel

            from legal_simplifier.training.rl_common import check_adapter_base

            check_adapter_base(self.spec.adapter, model_cfg)
            model = PeftModel.from_pretrained(model, str(self.spec.adapter)).merge_and_unload()
        import torch

        if torch.cuda.is_available():
            model = model.to("cuda")  # pyright: ignore[reportArgumentType]
        return model.eval(), tokenizer

    def generate(self, items: Sequence[EvalItem]) -> Sequence[str | None]:
        import torch

        model, tokenizer = self._load()
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        device = next(model.parameters()).device
        gen = self.config.generation
        outputs: list[str] = []
        try:
            for start in range(0, len(items), gen.batch_size):
                batch = items[start : start + gen.batch_size]
                texts = [
                    tokenizer.apply_chat_template(
                        build_prompt(item, self.examples),
                        tokenize=False,
                        add_generation_prompt=True,
                        **self.config.model.chat_template_kwargs,
                    )
                    for item in batch
                ]
                enc = tokenizer(
                    texts, return_tensors="pt", padding=True, add_special_tokens=False
                ).to(device)
                with torch.inference_mode():
                    generated = model.generate(
                        **enc,
                        do_sample=False,
                        max_new_tokens=gen.max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                decoded = tokenizer.batch_decode(
                    generated[:, enc["input_ids"].shape[1] :], skip_special_tokens=True
                )
                outputs.extend(completion_text(t) for t in decoded)
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return outputs


class ClaudeSystem(System):
    """Claude zero-shot via the Batches API (cached and cost-logged by ``BatchRunner``)."""

    paid = True

    def __init__(self, spec: SystemSpec, config: EvalConfigFile, *, client: Any | None = None):
        super().__init__(spec)
        self.config = config
        self.client = client
        self.last_report: dict[str, Any] | None = None

    def _identity(self) -> dict[str, Any]:
        c = self.config.claude
        return {"model": c.model, "effort": c.effort, "max_tokens": c.max_tokens}

    def request(self, item: EvalItem) -> dict[str, Any]:
        messages = build_messages(item.clause, item.category)
        c = self.config.claude
        return {
            "model": c.model,
            "max_tokens": c.max_tokens,
            "system": [
                {
                    "type": "text",
                    "text": messages[0]["content"],
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": messages[1]["content"]}],
            "output_config": {"effort": c.effort},
        }

    def runner(self) -> Any:
        from legal_simplifier.llm.batch import BatchRunner

        if self.client is None:
            import anthropic

            self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
        c = self.config.claude
        return BatchRunner(
            self.client.messages.batches,
            cache_dir=c.cache_dir,
            cost_log=c.cost_log,
            poll_interval_s=c.poll_interval_s,
        )

    def generate(self, items: Sequence[EvalItem]) -> Sequence[str | None]:
        from legal_simplifier.llm.batch import CachedResponse, request_key

        requests = [self.request(i) for i in items]
        results, report = self.runner().run(requests)
        self.last_report = report.to_dict()
        outputs: list[str | None] = []
        for params in requests:
            outcome = results.get(request_key(params))
            ok = isinstance(outcome, CachedResponse) and outcome.stop_reason == "end_turn"
            outputs.append(completion_text(outcome.text) if ok else None)  # type: ignore[union-attr]
        return outputs


def make_system(spec: SystemSpec, config: EvalConfigFile, *, client: Any | None = None) -> System:
    if spec.kind == "copy":
        return CopySystem(spec)
    if spec.kind == "truncate":
        return TruncateSystem(spec)
    if spec.kind == "hf":
        return HFSystem(spec, config)
    if spec.kind == "claude":
        return ClaudeSystem(spec, config, client=client)
    raise ValueError(f"unknown system kind {spec.kind!r}")


class GenerationCache:
    """``<root>/<system>/<fingerprint>.jsonl``: one ``{"key", "output"}`` line per item."""

    def __init__(self, root: Path | str):
        self.root = Path(root)

    def path(self, system: System) -> Path:
        return self.root / system.name / f"{system.fingerprint}.jsonl"

    def load(self, system: System) -> dict[str, str]:
        path = self.path(system)
        if not path.exists():
            return {}
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        return {r["key"]: r["output"] for r in rows}

    def add(self, system: System, outputs: dict[str, str]) -> None:
        path = self.path(system)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for key, output in outputs.items():
                handle.write(json.dumps({"key": key, "output": output}, ensure_ascii=False) + "\n")


def generate_cached(
    system: System, items: Sequence[EvalItem], cache: GenerationCache
) -> tuple[list[str], dict[str, int]]:
    """Outputs for ``items``, generating only the uncached ones.

    Returns the outputs (a failure is ``""``) and counts of cached, generated and
    failed outputs. Failures are not cached, so the next run retries them.
    """

    cached = cache.load(system)
    missing = [i for i in items if i.key not in cached]
    counts = {"cached": len(items) - len(missing), "generated": 0, "failed": 0}
    if missing:
        LOGGER.info("%s: generating %d of %d outputs", system.name, len(missing), len(items))
        fresh = dict(zip((i.key for i in missing), system.generate(missing), strict=True))
        ok = {k: v for k, v in fresh.items() if v is not None}
        cache.add(system, ok)
        cached.update(ok)
        counts["generated"], counts["failed"] = len(ok), len(fresh) - len(ok)
    return [cached.get(i.key, "") for i in items], counts
