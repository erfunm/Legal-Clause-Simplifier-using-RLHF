"""Advance cost range for a set of Message Batches requests, printed by ``--dry-run``.

Nothing here generates text or spends money. Input tokens come from one of
two counters, and the estimate always says which one was used:

* ``local``: a character-ratio approximation (2.5 to 4.0 characters per
  token). Claude's tokenizer is not available offline, so this gives a range,
  not a count.
* ``api``: Anthropic's token-counting endpoint (``messages.count_tokens``),
  which is free, sends no generations, and counts under the tokenizer of the
  request's model. It has its own requests-per-minute limit, so large runs
  count a deterministic sample and scale up the mean.

Output tokens cannot be counted in advance, because they include adaptive
thinking. The range runs from a floor (a rewrite about as long as the clause,
no thinking) to the hard ceiling (``max_tokens`` for every request; billed
output cannot exceed it). If the cost log already holds results for the same
model, for example from the pilot, the measured mean output per request is
reported as well, and that is the number to plan with.

Input pricing: the shared prefix (system prompt and output schema) is marked
for caching. The low end assumes one cache write and cache hits afterwards;
the high end assumes every request writes the cache (1.25x input). Batch
caching is best-effort, so the truth is somewhere in between.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol

from legal_simplifier.llm.batch import BATCH_DISCOUNT, PRICES_PER_MTOK

CounterName = Literal["local", "api"]

LOCAL_CHARS_PER_TOKEN = (4.0, 2.5)  # (fewest tokens, most tokens)
DEFAULT_API_SAMPLE = 100


class TokenCounter(Protocol):
    name: CounterName

    def count(self, params: Mapping[str, Any]) -> tuple[int, int]:
        """(low, high) input tokens for one request; equal for an exact counter."""
        ...


def _request_text(params: Mapping[str, Any]) -> str:
    """Every piece of text the model reads, for the local approximation."""

    parts: list[str] = []
    system = params.get("system")
    if isinstance(system, str):
        parts.append(system)
    elif system:
        parts.extend(block.get("text", "") for block in system)
    for message in params.get("messages", []):
        content = message["content"]
        if isinstance(content, str):
            parts.append(content)
        else:
            parts.extend(block.get("text", "") for block in content)
    if "output_config" in params:
        parts.append(json.dumps(params["output_config"].get("format", {})))
    return "\n".join(parts)


class LocalTokenCounter:
    name: CounterName = "local"

    def count(self, params: Mapping[str, Any]) -> tuple[int, int]:
        chars = len(_request_text(params))
        fewest, most = LOCAL_CHARS_PER_TOKEN
        return math.ceil(chars / fewest), math.ceil(chars / most)


class CountTokensAPI(Protocol):
    def count_tokens(self, **kwargs: Any) -> Any: ...


class ApiTokenCounter:
    """Anthropic's free token-counting endpoint. Never calls ``messages.create``."""

    name: CounterName = "api"

    def __init__(self, messages: CountTokensAPI):
        self.messages = messages

    def count(self, params: Mapping[str, Any]) -> tuple[int, int]:
        # count_tokens takes the create() parameters except max_tokens.
        kwargs = {k: v for k, v in params.items() if k != "max_tokens"}
        tokens = int(self.messages.count_tokens(**kwargs).input_tokens)
        return tokens, tokens


@dataclass
class CostEstimate:
    model: str
    n_requests: int
    counter: CounterName
    n_counted: int
    input_tokens: tuple[int, int]
    prefix_tokens: tuple[int, int]
    output_tokens: tuple[int, int]
    usd: tuple[float, float] | None
    measured_output_per_request: float | None = None
    measured_n: int = 0
    measured_usd: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _prefix_probe(params: Mapping[str, Any]) -> dict[str, Any]:
    """The request with its user message cut to one character: counts the shared prefix."""

    probe = dict(params)
    probe["messages"] = [{"role": "user", "content": "."}]
    return probe


def _clause_text(params: Mapping[str, Any]) -> str:
    content = params["messages"][-1]["content"]
    text = content if isinstance(content, str) else " ".join(b.get("text", "") for b in content)
    return text.split("Clause:\n", 1)[-1]


def _sample(requests: Sequence[Mapping[str, Any]], size: int) -> list[Mapping[str, Any]]:
    if len(requests) <= size:
        return list(requests)
    step = len(requests) / size
    return [requests[int(i * step)] for i in range(size)]


def estimate_cost(
    requests: Sequence[Mapping[str, Any]],
    counter: TokenCounter,
    *,
    sample_size: int = DEFAULT_API_SAMPLE,
    measured_output_tokens: Sequence[int] = (),
) -> CostEstimate | None:
    """Cost range for submitting ``requests`` as one batch run; ``None`` if there are none."""

    if not requests:
        return None
    model = str(requests[0]["model"])
    if any(r["model"] != model for r in requests):
        raise ValueError("estimate one model at a time")

    counted = _sample(requests, sample_size) if counter.name == "api" else list(requests)
    per_request = [counter.count(r) for r in counted]
    scale = len(requests) / len(counted)
    input_low = round(sum(lo for lo, _ in per_request) * scale)
    input_high = round(sum(hi for _, hi in per_request) * scale)
    prefix = counter.count(_prefix_probe(requests[0]))

    # Floor: the rewrite alone, about as many tokens as the clause (at ~4 characters per token).
    output_low = sum(math.ceil(len(_clause_text(r)) / LOCAL_CHARS_PER_TOKEN[0]) for r in requests)
    output_high = sum(int(r["max_tokens"]) for r in requests)

    estimate = CostEstimate(
        model=model,
        n_requests=len(requests),
        counter=counter.name,
        n_counted=len(counted),
        input_tokens=(input_low, input_high),
        prefix_tokens=prefix,
        output_tokens=(output_low, output_high),
        usd=None,
    )
    price = PRICES_PER_MTOK.get(model)
    if measured_output_tokens:
        estimate.measured_n = len(measured_output_tokens)
        estimate.measured_output_per_request = round(statistics.fmean(measured_output_tokens), 1)
    if price is None:
        return estimate

    n = len(requests)
    cached_low = prefix[0] * (n - 1)  # every request but the first reads the cached prefix
    low = (
        (input_low - cached_low) * price.input
        + cached_low * price.cache_read
        + output_low * price.output
    )
    high = input_high * price.cache_write + output_high * price.output
    estimate.usd = (
        round(low / 1e6 * BATCH_DISCOUNT, 6),
        round(high / 1e6 * BATCH_DISCOUNT, 6),
    )
    if estimate.measured_output_per_request is not None:
        mid_input = (input_low + input_high) / 2 * price.input
        mid_output = estimate.measured_output_per_request * n * price.output
        estimate.measured_usd = round((mid_input + mid_output) / 1e6 * BATCH_DISCOUNT, 6)
    return estimate


def _usd(value: float) -> str:
    return f"${value:,.2f}" if value >= 1 else f"${value:.4f}"


def describe(estimate: CostEstimate | Mapping[str, Any]) -> str:
    """Human-readable lines for the CLI."""

    e = estimate.to_dict() if isinstance(estimate, CostEstimate) else dict(estimate)
    if e["counter"] == "api":
        how = (
            "Anthropic token-counting endpoint (free, no generations), "
            f"{e['n_counted']} of {e['n_requests']} requests counted"
            + (" and scaled up" if e["n_counted"] < e["n_requests"] else "")
        )
    else:
        fewest, most = LOCAL_CHARS_PER_TOKEN
        how = (
            f"local approximation ({most}-{fewest} characters per token; "
            "Claude's tokenizer is not available offline; use --count-tokens api for counts)"
        )
    lo_in, hi_in = e["input_tokens"]
    lo_out, hi_out = e["output_tokens"]
    lines = [
        f"Estimate for {e['n_requests']} requests to {e['model']} (Batch API prices):",
        f"  input tokens:  {lo_in:,}" + ("" if lo_in == hi_in else f" to {hi_in:,}") + f"  [{how}]",
        f"  output tokens: {lo_out:,} (rewrite only, no thinking) to {hi_out:,} "
        "(max_tokens on every request: the hard ceiling)",
    ]
    if e["usd"] is None:
        lines.append(f"  cost: unknown (no price for {e['model']} in PRICES_PER_MTOK)")
    else:
        lines.append(f"  cost range: {_usd(e['usd'][0])} to {_usd(e['usd'][1])}")
    if e.get("measured_output_per_request") is not None:
        lines.append(
            f"  measured output so far: {e['measured_output_per_request']:,.0f} tokens per request "
            f"over {e['measured_n']} billed results"
            + (
                f" -> about {_usd(e['measured_usd'])} for this run"
                if e.get("measured_usd") is not None
                else ""
            )
        )
    else:
        lines.append("  no measured output yet: run a pilot (--limit 20 --yes) to narrow the range")
    return "\n".join(lines)
