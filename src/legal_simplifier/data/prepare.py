"""Turn clauses into deduplicated, contract-level train/dev/test files.

Two entry points share the same dedup -> split -> leakage-check core:

* :func:`prepare_from_cuad` extracts clause candidates from CUAD and writes
  ``candidates.jsonl`` (no references yet; references are generated next, so
  no API spend goes on near-duplicates).
* :func:`prepare_from_records` takes records that already have references
  (the test fixtures) and writes ``train/dev/test.jsonl`` directly.

Every filter's thresholds and counts go to ``stats.json``, which the dataset
card is rendered from.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

from legal_simplifier.data.cuad import (
    CUAD_FILE,
    CUAD_REPO,
    CUAD_REVISION,
    DEFAULT_CATEGORIES,
    download_cuad,
    extract_clauses,
    load_cuad_json,
)
from legal_simplifier.data.schema import (
    ClauseCandidate,
    ClauseRecord,
    read_records,
    write_records,
)
from legal_simplifier.data.splitting import (
    SPLITS,
    DedupReport,
    check_no_leakage,
    deduplicate,
    split_by_contract,
    split_counts,
)

LOGGER = logging.getLogger(__name__)

STATS_FILE = "stats.json"
CANDIDATES_FILE = "candidates.jsonl"
REMOVED_DUPLICATES_FILE = "removed_near_duplicates.jsonl"

C = TypeVar("C", bound=ClauseCandidate)


@dataclass
class PrepareResult(Generic[C]):
    records: list[C]
    stats: dict[str, Any]
    output_dir: Path


def read_stats(output_dir: Path | str) -> dict[str, Any]:
    path = Path(output_dir) / STATS_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_stats(output_dir: Path | str, stats: dict[str, Any]) -> Path:
    path = Path(output_dir) / STATS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _dedup_and_split(
    items: Sequence[C],
    output_dir: Path,
    *,
    dedup_threshold: float,
    ratios: tuple[float, float, float],
    seed: int,
) -> tuple[list[C], DedupReport]:
    kept, dedup = deduplicate(items, threshold=dedup_threshold)
    split = split_by_contract(kept, ratios=ratios, seed=seed)
    problems = check_no_leakage(split, threshold=dedup_threshold)
    if problems:
        raise RuntimeError("split leakage detected:\n" + "\n".join(problems[:20]))

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / REMOVED_DUPLICATES_FILE).write_text(
        "".join(
            json.dumps({"removed": a, "duplicate_of": b, "jaccard": s}) + "\n"
            for a, b, s in dedup.removed
        ),
        encoding="utf-8",
    )
    LOGGER.info(
        "Kept %d of %d clauses (%d near-duplicates removed); splits: %s",
        dedup.n_kept,
        dedup.n_input,
        dedup.n_removed,
        split_counts(split),
    )
    return split, dedup


def _split_stats(
    split: Sequence[ClauseCandidate], ratios: tuple[float, float, float], seed: int
) -> dict[str, Any]:
    return {
        "seed": seed,
        "ratios": list(ratios),
        "splits": split_counts(split),
        "leakage_problems": 0,
    }


def prepare_from_cuad(
    output_dir: Path | str,
    *,
    cuad_json: Path | str | None = None,
    categories: Sequence[str] = DEFAULT_CATEGORIES,
    min_chars: int = 40,
    min_words: int = 6,
    max_chars: int = 1500,
    dedup_threshold: float = 0.8,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 13,
) -> PrepareResult[ClauseCandidate]:
    """Extract, filter, deduplicate and split CUAD clauses; write ``candidates.jsonl``.

    ``cuad_json`` defaults to the pinned ``CUAD_v1.json`` from the Hub.
    """

    output_dir = Path(output_dir)
    source_path = Path(cuad_json) if cuad_json else download_cuad()
    candidates, extraction = extract_clauses(
        load_cuad_json(source_path),
        categories=categories,
        min_chars=min_chars,
        min_words=min_words,
        max_chars=max_chars,
    )
    split, dedup = _dedup_and_split(
        candidates, output_dir, dedup_threshold=dedup_threshold, ratios=ratios, seed=seed
    )
    write_records(split, output_dir / CANDIDATES_FILE)

    stats: dict[str, Any] = {
        "source": {
            "dataset": CUAD_REPO if cuad_json is None else str(source_path),
            "file": CUAD_FILE,
            "revision": CUAD_REVISION if cuad_json is None else None,
            "license": "CC BY 4.0",
        },
        "filters": {
            "extraction": extraction.to_dict(),
            "near_duplicates": dedup.to_dict(),
        },
        "filtered_examples": extraction.filtered_examples,
        **_split_stats(split, ratios, seed),
        "kept_per_category": _per_category(split),
    }
    write_stats(output_dir, stats)
    return PrepareResult(records=split, stats=stats, output_dir=output_dir)


def _per_category(items: Sequence[ClauseCandidate]) -> dict[str, dict[str, int]]:
    table: dict[str, dict[str, int]] = {}
    for item in items:
        row = table.setdefault(item.category, dict.fromkeys(SPLITS, 0))
        if item.split:
            row[item.split] += 1
    return dict(sorted(table.items()))


def reference_source_counts(records: Sequence[ClauseRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.reference_source] = counts.get(record.reference_source, 0) + 1
    return counts


def write_split_files(records: Sequence[ClauseRecord], output_dir: Path | str) -> None:
    for name in SPLITS:
        write_records((r for r in records if r.split == name), Path(output_dir) / f"{name}.jsonl")


def prepare_from_records(
    records: list[ClauseRecord],
    output_dir: Path | str,
    *,
    dedup_threshold: float = 0.8,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 13,
) -> PrepareResult[ClauseRecord]:
    """Deduplicate and split records that already have references; write the split files."""

    output_dir = Path(output_dir)
    split, dedup = _dedup_and_split(
        records, output_dir, dedup_threshold=dedup_threshold, ratios=ratios, seed=seed
    )
    write_split_files(split, output_dir)
    stats: dict[str, Any] = {
        "filters": {"near_duplicates": dedup.to_dict()},
        **_split_stats(split, ratios, seed),
        "reference_source": reference_source_counts(split),
    }
    write_stats(output_dir, stats)
    return PrepareResult(records=split, stats=stats, output_dir=output_dir)


def prepare_from_jsonl(
    input_path: Path | str, output_dir: Path | str, **kwargs: Any
) -> PrepareResult[ClauseRecord]:
    return prepare_from_records(read_records(input_path), output_dir, **kwargs)
