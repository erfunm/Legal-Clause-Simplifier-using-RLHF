"""Checks that stop mock or smoke-test output from being published as results.

Anything that writes results tables, figures or README sections must call
:func:`assert_publishable_run` first. A run is publishable only when it is not
a smoke run and every example's reference came from Claude or a human.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from legal_simplifier.data.schema import ClauseRecord

PUBLISHABLE_SOURCES = frozenset({"claude", "human"})
MANIFEST_FILE = "manifest.json"
EXAMPLES_FILE = "examples.jsonl"


class UnpublishableResultsError(RuntimeError):
    """Raised when results rest on mock references or come from a smoke run."""


def _source_of(record: ClauseRecord | Mapping[str, Any]) -> object:
    if isinstance(record, ClauseRecord):
        return record.reference_source
    return record.get("reference_source")


def assert_publishable_records(records: Iterable[ClauseRecord | Mapping[str, Any]]) -> int:
    """Raise unless every record has a Claude or human reference. Returns the record count."""

    counts: Counter[str] = Counter()
    for record in records:
        counts[str(_source_of(record))] += 1
    if not counts:
        raise UnpublishableResultsError("no examples to report on")
    bad = {source: n for source, n in counts.items() if source not in PUBLISHABLE_SOURCES}
    if bad:
        raise UnpublishableResultsError(
            f"refusing to publish: {sum(bad.values())} of {counts.total()} examples have "
            f"non-publishable reference_source {bad}. Generate real references first "
            "(see docs/RUNBOOK.md)."
        )
    return counts.total()


def assert_publishable_run(run_dir: Path | str) -> None:
    """Raise unless ``run_dir`` holds a non-smoke run whose references are all real."""

    run_dir = Path(run_dir)
    manifest_path = run_dir / MANIFEST_FILE
    examples_path = run_dir / EXAMPLES_FILE
    if not manifest_path.exists() or not examples_path.exists():
        raise UnpublishableResultsError(
            f"{run_dir} is missing {MANIFEST_FILE} or {EXAMPLES_FILE}; not a complete eval run"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("smoke", True):
        raise UnpublishableResultsError(
            f"refusing to publish {run_dir}: it is a smoke run (or does not say otherwise)"
        )
    with examples_path.open(encoding="utf-8") as handle:
        assert_publishable_records(json.loads(line) for line in handle if line.strip())
