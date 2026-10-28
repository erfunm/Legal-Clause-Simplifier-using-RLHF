"""Record schema shared by every pipeline stage.

Every clause record says where its reference simplification came from
(``reference_source``). Mock references exist so the pipeline can run offline
and in CI; they must never end up in published results (see
:mod:`legal_simplifier.eval.guards`).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ReferenceSource = Literal["mock", "claude", "human"]
Split = Literal["train", "dev", "test"]


class ClauseRecord(BaseModel):
    """One legal clause with its plain-English reference and required entities."""

    model_config = ConfigDict(extra="forbid")

    id: str
    contract_id: str = Field(description="Source contract; splits never share a contract.")
    category: str
    clause: str
    reference: str
    reference_source: ReferenceSource
    entities: list[str] = Field(default_factory=list)
    split: Split | None = None


def read_records(path: Path | str) -> list[ClauseRecord]:
    """Load records from a JSON Lines file."""

    return list(iter_records(path))


def iter_records(path: Path | str) -> Iterator[ClauseRecord]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield ClauseRecord.model_validate_json(line)


def write_records(records: Iterable[ClauseRecord], path: Path | str) -> Path:
    """Write records to a JSON Lines file, creating parent directories."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")
    return path
