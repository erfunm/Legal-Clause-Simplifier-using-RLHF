"""Silver plain-English references for clause candidates.

Backends:

* ``mock`` (default, offline): the reference is the clause itself, marked
  ``reference_source="mock"``. It only exists so the pipeline runs in CI;
  the publishing guards refuse anything built on it.
* ``claude``: ``claude-sonnet-5-5`` through the Message Batches API with the
  style guide below, structured JSON output, on-disk caching and a cost log
  (see :mod:`legal_simplifier.llm.batch`).

Each generated reference is checked against the clause's required entities;
any it drops are listed in ``reference_issues.jsonl`` for the human review.

A pilot (``limit=N``) runs a category-balanced sample of N candidates and
writes ``pilot_references.jsonl`` and ``pilot_issues.jsonl`` instead of the
split files, so it can be read before the full run. Its responses land in the
same cache, so the full run does not pay for them again.
"""

from __future__ import annotations

import json
import logging
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from legal_simplifier.config import ReferenceConfig
from legal_simplifier.data.entities import check_entity_presence
from legal_simplifier.data.prepare import (
    CANDIDATES_FILE,
    read_stats,
    reference_source_counts,
    write_split_files,
    write_stats,
)
from legal_simplifier.data.sampling import balanced_sample
from legal_simplifier.data.schema import ClauseCandidate, ClauseRecord, read_jsonl, write_records
from legal_simplifier.llm.batch import (
    BatchesAPI,
    BatchRunner,
    CachedResponse,
    Failure,
    RunReport,
    request_key,
)
from legal_simplifier.llm.estimate import (
    DEFAULT_API_SAMPLE,
    ApiTokenCounter,
    CounterName,
    LocalTokenCounter,
    TokenCounter,
    estimate_cost,
)

LOGGER = logging.getLogger(__name__)

ISSUES_FILE = "reference_issues.jsonl"
PILOT_FILE = "pilot_references.jsonl"
PILOT_ISSUES_FILE = "pilot_issues.jsonl"

STYLE_GUIDE = """\
You rewrite clauses from commercial contracts into plain English for a reader with no legal training.

Requirements, in priority order:
1. Keep the legal meaning exactly. Every obligation stays an obligation ("must"), every permission stays a permission ("may" or "can"), and every prohibition stays a prohibition ("must not"). Keep every condition, exception and time limit.
2. Keep every party name, defined term, date, amount of money, percentage, duration and number. Write numbers as digits ("30 days", "$250,000"). Keep defined terms with the same capitalisation as the clause (for example "the Licensee", "the Territory").
3. Add nothing: no new facts, no examples, no advice, no commentary, no guesses about the rest of the contract.
4. Make it easy to read: aim for a US grade 8 reading level, short sentences (about 20 words or fewer), everyday words, active voice. Replace legal phrasing ("hereinafter", "notwithstanding", "shall") with plain wording. Use a short list only if the clause itself lists several items.
5. If the clause is an incomplete fragment, simplify only what is there. Do not complete it.
6. Remove nothing except redundant legal phrasing. If text is redacted (for example "[*]"), keep the redaction marker.

Return JSON with one field, "simplified", holding only the rewritten clause."""

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"simplified": {"type": "string"}},
    "required": ["simplified"],
    "additionalProperties": False,
}


def build_request(candidate: ClauseCandidate, config: ReferenceConfig) -> dict[str, Any]:
    """Messages API parameters for one candidate (deterministic, so they key the cache)."""

    entity_line = (
        "Entities that must appear in the rewrite: " + "; ".join(candidate.entities)
        if candidate.entities
        else "No specific entities were detected; still keep every name, date, amount and defined term."
    )
    return {
        "model": config.model,
        "max_tokens": config.max_tokens,
        "system": [{"type": "text", "text": STYLE_GUIDE, "cache_control": {"type": "ephemeral"}}],
        "messages": [
            {
                "role": "user",
                "content": f"Clause type: {candidate.category}\n{entity_line}\n\nClause:\n{candidate.clause}",
            }
        ],
        "output_config": {
            "effort": config.effort,
            "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA},
        },
    }


def parse_response(response: CachedResponse) -> str:
    """Return the simplified text, or raise ``ValueError`` if the response is unusable."""

    if response.stop_reason != "end_turn":
        raise ValueError(f"stop_reason={response.stop_reason!r}")
    try:
        simplified = json.loads(response.text)["simplified"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"malformed JSON output: {response.text[:200]!r}") from exc
    if not isinstance(simplified, str) or not simplified.strip():
        raise ValueError("empty simplification")
    return " ".join(simplified.split())


@dataclass
class ReferenceReport:
    backend: str
    model: str | None
    n_candidates: int = 0
    n_with_reference: int = 0
    failures: dict[str, int] = field(default_factory=dict)
    batch: dict[str, Any] | None = None
    entity_recall_mean: float | None = None
    n_missing_entities: int = 0
    estimate: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.model,
            "n_candidates": self.n_candidates,
            "n_with_reference": self.n_with_reference,
            "failures": self.failures,
            "batch": self.batch,
            "reference_entity_recall_mean": self.entity_recall_mean,
            "n_references_missing_entities": self.n_missing_entities,
        }


class ReferenceGenerator(Protocol):
    def generate(
        self, candidates: Sequence[ClauseCandidate]
    ) -> tuple[list[ClauseRecord], ReferenceReport]: ...


class MockReferenceGenerator:
    """Copies the clause as its own "reference" so offline runs work. Never publishable."""

    def generate(
        self, candidates: Sequence[ClauseCandidate]
    ) -> tuple[list[ClauseRecord], ReferenceReport]:
        records = [
            ClauseRecord(**c.model_dump(), reference=c.clause, reference_source="mock")
            for c in candidates
        ]
        report = ReferenceReport("mock", None, len(candidates), len(records))
        return records, report


class ClaudeReferenceGenerator:
    def __init__(
        self,
        runner: BatchRunner,
        config: ReferenceConfig,
        *,
        dry_run: bool = False,
        counter: TokenCounter | None = None,
        count_sample: int = DEFAULT_API_SAMPLE,
    ):
        self.runner = runner
        self.config = config
        self.dry_run = dry_run
        self.counter = counter or LocalTokenCounter()
        self.count_sample = count_sample

    def estimate(self, requests: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
        """Cost range for the requests a real run would submit (cached ones are free)."""

        measured = [
            int(e["usage"].get("output_tokens") or 0)
            for e in self.runner.cost_log.entries()
            if e.get("model") == self.config.model and e.get("usage")
        ]
        estimate = estimate_cost(
            self.runner.unsubmitted(requests),
            self.counter,
            sample_size=self.count_sample,
            measured_output_tokens=measured,
        )
        return estimate.to_dict() if estimate else None

    def generate(
        self, candidates: Sequence[ClauseCandidate]
    ) -> tuple[list[ClauseRecord], ReferenceReport]:
        requests = [build_request(c, self.config) for c in candidates]
        results, run_report = self.runner.run(requests, dry_run=self.dry_run)
        report = ReferenceReport("claude", self.config.model, len(candidates))
        report.batch = run_report.to_dict()
        if self.dry_run:
            report.estimate = self.estimate(requests)
        records: list[ClauseRecord] = []
        failures: dict[str, int] = dict(run_report.failures) if not self.dry_run else {}
        for candidate, params in zip(candidates, requests, strict=True):
            outcome = results.get(request_key(params))
            if outcome is None or isinstance(outcome, Failure):
                continue
            try:
                text = parse_response(outcome)
            except ValueError as exc:
                LOGGER.warning("Unusable reference for %s: %s", candidate.id, exc)
                failures["unparseable"] = failures.get("unparseable", 0) + 1
                continue
            records.append(
                ClauseRecord(**candidate.model_dump(), reference=text, reference_source="claude")
            )
        report.failures = failures
        report.n_with_reference = len(records)
        return records, report


def entity_issues(records: Sequence[ClauseRecord]) -> list[dict[str, Any]]:
    """Records whose reference drops required entities, for targeted human review."""

    issues = []
    for record in records:
        presence = check_entity_presence(record.entities, record.reference)
        missing = [e for e, kept in presence.items() if not kept]
        if missing:
            issues.append({"id": record.id, "split": record.split, "missing_entities": missing})
    return issues


def _entity_recalls(records: Sequence[ClauseRecord]) -> list[float]:
    recalls = []
    for record in records:
        presence = check_entity_presence(record.entities, record.reference)
        if presence:
            recalls.append(sum(presence.values()) / len(presence))
    return recalls


def make_generator(
    config: ReferenceConfig,
    *,
    dry_run: bool = False,
    client: Any | None = None,
    count_tokens: CounterName = "local",
) -> ReferenceGenerator:
    if config.backend == "mock":
        return MockReferenceGenerator()
    if client is None:
        import anthropic

        client = (
            anthropic.Anthropic()
        )  # reads ANTHROPIC_API_KEY / auth profile from the environment
    batches: BatchesAPI = client.messages.batches
    runner = BatchRunner(
        batches,
        cache_dir=config.cache_dir,
        cost_log=config.cost_log,
        poll_interval_s=config.poll_interval_s,
    )
    counter: TokenCounter = (
        ApiTokenCounter(client.messages) if count_tokens == "api" else LocalTokenCounter()
    )
    return ClaudeReferenceGenerator(runner, config, dry_run=dry_run, counter=counter)


def select_pilot(
    candidates: Sequence[ClauseCandidate], limit: int, *, seed: int
) -> list[ClauseCandidate]:
    """A category-balanced, seeded sample of ``limit`` candidates for a paid pilot."""

    if limit < 1:
        raise ValueError("limit must be at least 1")
    return balanced_sample(candidates, limit, seed=seed)


def generate_references(
    data_dir: Path | str,
    generator: ReferenceGenerator,
    *,
    limit: int | None = None,
    seed: int = 21,
) -> tuple[list[ClauseRecord], ReferenceReport]:
    """Read ``candidates.jsonl``, attach references, write split files and update ``stats.json``.

    With ``limit`` only a balanced pilot sample is processed, and the results go
    to ``pilot_references.jsonl`` (plus ``pilot_issues.jsonl`` and a ``pilot``
    section in the stats) so the split files stay untouched.
    """

    data_dir = Path(data_dir)
    candidates = read_jsonl(data_dir / CANDIDATES_FILE, ClauseCandidate)
    if limit is not None:
        candidates = select_pilot(candidates, limit, seed=seed)
    records, report = generator.generate(candidates)
    if isinstance(generator, ClaudeReferenceGenerator) and generator.dry_run:
        return records, report

    recalls = _entity_recalls(records)
    report.entity_recall_mean = round(statistics.fmean(recalls), 4) if recalls else None
    issues = entity_issues(records)
    report.n_missing_entities = len(issues)

    if limit is not None:
        write_records(records, data_dir / PILOT_FILE)
        _write_issues(data_dir / PILOT_ISSUES_FILE, issues)
        stats = read_stats(data_dir)
        stats["pilot"] = {"limit": limit, "seed": seed, **report.to_dict()}
        write_stats(data_dir, stats)
        LOGGER.info("Pilot: %d references in %s", len(records), data_dir / PILOT_FILE)
        return records, report

    write_split_files(records, data_dir)
    _write_issues(data_dir / ISSUES_FILE, issues)
    stats = read_stats(data_dir)
    stats["references"] = report.to_dict()
    stats["reference_source"] = reference_source_counts(records)
    write_stats(data_dir, stats)
    LOGGER.info(
        "References: %d of %d candidates (%s); %d drop required entities",
        report.n_with_reference,
        report.n_candidates,
        report.backend,
        report.n_missing_entities,
    )
    return records, report


def _write_issues(path: Path, issues: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(dict(i), ensure_ascii=False) + "\n" for i in issues), encoding="utf-8"
    )


def summarise_run(report: RunReport | Mapping[str, Any]) -> str:
    data = report.to_dict() if isinstance(report, RunReport) else dict(report)
    return (
        f"{data['n_unique']} unique requests: {data['n_cached']} cached, "
        f"{data['n_submitted']} {'would be ' if data['dry_run'] else ''}submitted"
    )
