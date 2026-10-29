"""Run Claude requests through the Message Batches API, with an on-disk cache and a cost log.

Used for the silver references, the Claude evaluation system and the LLM judge.

* **Cache:** every successful response is stored under the SHA-256 of its
  request parameters, so re-running a stage costs nothing.
* **Resume:** submitted batch ids are recorded before polling, so if the
  process dies the next run collects those results instead of paying for
  the same requests again.
* **Cost log:** one JSON line per billed result with token usage and the
  cost at list batch prices (``PRICES_PER_MTOK``). It records spend that
  happened; advance estimates live in :mod:`legal_simplifier.llm.estimate`.
* **Failures** (errored, expired, canceled, refused) are returned and
  counted, never cached, and never replaced with a fallback text.

The client is passed in, so tests drive this with a fake that replays
recorded responses and the real ``anthropic.Anthropic()`` is only built by
the CLI.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Price:
    """USD per million tokens at standard (non-batch) rates."""

    input: float
    output: float
    cache_read: float

    @property
    def cache_write(self) -> float:
        return self.input * 1.25  # 5-minute cache writes


# Standard list prices, checked on 2026-10-28 against
#   https://platform.claude.com/docs/en/about-claude/pricing
# ("Model pricing" table). Cache hits on Sonnet 5.5 and Opus 5.5 are 0.05x the
# base input price; 5-minute cache writes are 1.25x. The Batch API halves input
# and output ("Batch processing": Sonnet 5.5 $1 / $5, Opus 5.5 $2 / $10 per
# MTok), and the page says the cache multipliers stack with that discount.
PRICES_PER_MTOK: dict[str, Price] = {
    "claude-sonnet-5-5": Price(input=2.00, output=10.00, cache_read=0.10),
    "claude-opus-5-5": Price(input=4.00, output=20.00, cache_read=0.20),
}
BATCH_DISCOUNT = 0.5
MAX_REQUESTS_PER_BATCH = 10_000


def cost_usd(model: str, usage: Mapping[str, Any], *, batch: bool = True) -> float | None:
    """Cost of one response from its ``usage``; ``None`` for a model without a known price."""

    price = PRICES_PER_MTOK.get(model)
    if price is None:
        return None
    dollars = (
        (usage.get("input_tokens") or 0) * price.input
        + (usage.get("output_tokens") or 0) * price.output
        + (usage.get("cache_read_input_tokens") or 0) * price.cache_read
        + (usage.get("cache_creation_input_tokens") or 0) * price.cache_write
    ) / 1_000_000
    return dollars * (BATCH_DISCOUNT if batch else 1.0)


def request_key(params: Mapping[str, Any]) -> str:
    """Stable 64-hex-char key for a request; also valid as a batch ``custom_id``."""

    canonical = json.dumps(params, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class CachedResponse:
    key: str
    model: str
    text: str
    stop_reason: str | None
    usage: dict[str, Any]
    batch_id: str | None
    created_at: str


@dataclass
class Failure:
    key: str
    kind: str  # errored | expired | canceled | refusal | missing
    detail: str = ""


class ResponseCache:
    """One JSON file per successful response, sharded by key prefix."""

    def __init__(self, root: Path | str):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> CachedResponse | None:
        path = self._path(key)
        if not path.exists():
            return None
        return CachedResponse(**json.loads(path.read_text(encoding="utf-8")))

    def put(self, response: CachedResponse) -> None:
        path = self._path(response.key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(response), ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)


class CostLog:
    """Append-only JSON Lines log of billed results."""

    def __init__(self, path: Path | str):
        self.path = Path(path)

    def append(self, entry: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(entry), ensure_ascii=False) + "\n")

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def total_usd(self) -> float:
        return sum(e.get("cost_usd") or 0.0 for e in self.entries())


class BatchesAPI(Protocol):
    """The subset of ``anthropic.Anthropic().messages.batches`` this module uses."""

    def create(self, *, requests: Any) -> Any: ...
    def retrieve(self, message_batch_id: str) -> Any: ...
    def results(self, message_batch_id: str) -> Any: ...


@dataclass
class RunReport:
    n_requested: int = 0
    n_unique: int = 0
    n_cached: int = 0
    n_submitted: int = 0
    n_succeeded: int = 0
    failures: dict[str, int] = field(default_factory=dict)
    batch_ids: list[str] = field(default_factory=list)
    cost_usd: float = 0.0
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _as_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, Mapping):
        return dict(obj)
    return obj.model_dump(mode="json", exclude_none=True)


class BatchRunner:
    def __init__(
        self,
        batches: BatchesAPI,
        *,
        cache_dir: Path | str,
        cost_log: Path | str,
        poll_interval_s: float = 60.0,
        max_requests_per_batch: int = MAX_REQUESTS_PER_BATCH,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.batches = batches
        self.cache = ResponseCache(cache_dir)
        self.cost_log = CostLog(cost_log)
        self.pending_path = Path(cache_dir) / "pending_batches.jsonl"
        self.poll_interval_s = poll_interval_s
        self.max_requests_per_batch = max_requests_per_batch
        self.sleep = sleep

    # -- pending-batch bookkeeping -------------------------------------------

    def _pending(self) -> list[dict[str, Any]]:
        if not self.pending_path.exists():
            return []
        with self.pending_path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def _write_pending(self, entries: Sequence[Mapping[str, Any]]) -> None:
        self.pending_path.parent.mkdir(parents=True, exist_ok=True)
        self.pending_path.write_text(
            "".join(json.dumps(dict(e)) + "\n" for e in entries), encoding="utf-8"
        )

    # -- main entry point ----------------------------------------------------

    def run(
        self, requests: Sequence[Mapping[str, Any]], *, dry_run: bool = False
    ) -> tuple[dict[str, CachedResponse | Failure], RunReport]:
        """Return a result per request key, submitting only what is not cached or pending.

        With ``dry_run`` nothing is sent: the report says how many requests
        would be submitted.
        """

        report = RunReport(n_requested=len(requests), dry_run=dry_run)
        by_key: dict[str, Mapping[str, Any]] = {}
        for params in requests:
            by_key.setdefault(request_key(params), params)
        report.n_unique = len(by_key)

        results, pending, to_submit = self._plan(by_key)
        report.n_cached = len(results)
        report.n_submitted = len(to_submit)
        if dry_run:
            return results, report

        for start in range(0, len(to_submit), self.max_requests_per_batch):
            chunk = to_submit[start : start + self.max_requests_per_batch]
            batch = self.batches.create(
                requests=[{"custom_id": key, "params": dict(by_key[key])} for key in chunk]
            )
            entry = {"batch_id": batch.id, "custom_ids": chunk, "created_at": _now()}
            pending.append(entry)
            self._write_pending([*self._pending(), entry])
            LOGGER.info("Submitted batch %s with %d requests", batch.id, len(chunk))

        for entry in pending:
            batch_id = entry["batch_id"]
            report.batch_ids.append(batch_id)
            self._wait(batch_id)
            for key, outcome in self._collect(batch_id, report).items():
                if key in by_key:
                    results[key] = outcome
            self._write_pending([p for p in self._pending() if p["batch_id"] != batch_id])

        for key in by_key:
            if key not in results:
                results[key] = Failure(key, "missing", "no result returned for this request")
        report.n_succeeded = sum(isinstance(r, CachedResponse) for r in results.values())
        for outcome in results.values():
            if isinstance(outcome, Failure):
                report.failures[outcome.kind] = report.failures.get(outcome.kind, 0) + 1
        return results, report

    def unsubmitted(self, requests: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        """The unique requests a real run would submit: neither cached nor in a pending batch."""

        by_key: dict[str, Mapping[str, Any]] = {}
        for params in requests:
            by_key.setdefault(request_key(params), params)
        return [by_key[key] for key in self._plan(by_key)[2]]

    def _plan(
        self, by_key: Mapping[str, Mapping[str, Any]]
    ) -> tuple[dict[str, CachedResponse | Failure], list[dict[str, Any]], list[str]]:
        results: dict[str, CachedResponse | Failure] = {}
        for key in by_key:
            cached = self.cache.get(key)
            if cached is not None:
                results[key] = cached
        pending = [
            p for p in self._pending() if set(p["custom_ids"]) & (set(by_key) - set(results))
        ]
        pending_ids = {cid for p in pending for cid in p["custom_ids"]}
        to_submit = [k for k in by_key if k not in results and k not in pending_ids]
        return results, pending, to_submit

    def _wait(self, batch_id: str) -> None:
        while True:
            batch = self.batches.retrieve(batch_id)
            if batch.processing_status == "ended":
                return
            LOGGER.info("Batch %s: %s", batch_id, batch.processing_status)
            self.sleep(self.poll_interval_s)

    def _collect(self, batch_id: str, report: RunReport) -> dict[str, CachedResponse | Failure]:
        outcomes: dict[str, CachedResponse | Failure] = {}
        for item in self.batches.results(batch_id):
            key = item.custom_id
            result = item.result
            if result.type != "succeeded":
                detail = _as_dict(getattr(result, "error", None))
                outcomes[key] = Failure(key, result.type, json.dumps(detail)[:500])
                continue
            message = result.message
            usage = _as_dict(message.usage)
            cost = cost_usd(message.model, usage)
            report.cost_usd += cost or 0.0
            self.cost_log.append(
                {
                    "time": _now(),
                    "batch_id": batch_id,
                    "custom_id": key,
                    "model": message.model,
                    "stop_reason": message.stop_reason,
                    "usage": usage,
                    "cost_usd": cost,
                }
            )
            if message.stop_reason == "refusal":
                details = _as_dict(getattr(message, "stop_details", None))
                outcomes[key] = Failure(key, "refusal", json.dumps(details)[:500])
                continue
            text = "".join(b.text for b in message.content if b.type == "text")
            response = CachedResponse(
                key=key,
                model=message.model,
                text=text,
                stop_reason=message.stop_reason,
                usage=usage,
                batch_id=batch_id,
                created_at=_now(),
            )
            self.cache.put(response)
            outcomes[key] = response
        return outcomes
