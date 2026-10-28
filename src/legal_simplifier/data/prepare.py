"""Turn clause records into deduplicated, contract-level train/dev/test files.

Phase 0 reads records from a JSON Lines file (the test fixtures use this).
CUAD extraction and reference generation plug in ahead of this step.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from legal_simplifier.data.schema import ClauseRecord, read_records, write_records
from legal_simplifier.data.splitting import (
    SPLITS,
    check_no_leakage,
    deduplicate,
    split_by_contract,
    split_counts,
)

LOGGER = logging.getLogger(__name__)

STATS_FILE = "stats.json"


@dataclass
class PrepareResult:
    records: list[ClauseRecord]
    stats: dict[str, Any]
    output_dir: Path


def prepare_from_records(
    records: list[ClauseRecord],
    output_dir: Path | str,
    *,
    dedup_threshold: float = 0.8,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 13,
) -> PrepareResult:
    """Deduplicate, split by contract, verify no leakage and write the split files + stats."""

    output_dir = Path(output_dir)
    kept, dedup = deduplicate(records, threshold=dedup_threshold)
    split = split_by_contract(kept, ratios=ratios, seed=seed)
    problems = check_no_leakage(split, threshold=dedup_threshold)
    if problems:
        raise RuntimeError("split leakage detected:\n" + "\n".join(problems[:20]))

    for name in SPLITS:
        write_records((r for r in split if r.split == name), output_dir / f"{name}.jsonl")

    sources: dict[str, int] = {}
    for record in split:
        sources[record.reference_source] = sources.get(record.reference_source, 0) + 1
    stats: dict[str, Any] = {
        "seed": seed,
        "ratios": list(ratios),
        "filters": {"near_duplicates": dedup.to_dict()},
        "splits": split_counts(split),
        "reference_source": sources,
        "leakage_problems": 0,
    }
    (output_dir / STATS_FILE).write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    (output_dir / "removed_near_duplicates.jsonl").write_text(
        "".join(
            json.dumps({"removed": a, "duplicate_of": b, "jaccard": s}) + "\n"
            for a, b, s in dedup.removed
        ),
        encoding="utf-8",
    )
    LOGGER.info(
        "Kept %d of %d records (%d near-duplicates removed); splits: %s",
        dedup.n_kept,
        dedup.n_input,
        dedup.n_removed,
        stats["splits"],
    )
    return PrepareResult(records=split, stats=stats, output_dir=output_dir)


def prepare_from_jsonl(
    input_path: Path | str, output_dir: Path | str, **kwargs: Any
) -> PrepareResult:
    return prepare_from_records(read_records(input_path), output_dir, **kwargs)
