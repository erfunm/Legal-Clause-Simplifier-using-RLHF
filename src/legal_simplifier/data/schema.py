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
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

ReferenceSource = Literal["mock", "claude", "human"]
Split = Literal["train", "dev", "test"]


class ClauseCandidate(BaseModel):
    """An extracted clause and its required entities, before it has a reference."""

    model_config = ConfigDict(extra="forbid")

    id: str
    contract_id: str = Field(description="Source contract; splits never share a contract.")
    category: str
    extra_categories: list[str] = Field(
        default_factory=list,
        description="Other categories CUAD gave the same span in the same contract.",
    )
    clause: str
    entities: list[str] = Field(default_factory=list)
    split: Split | None = None


class ClauseRecord(ClauseCandidate):
    """A clause with its plain-English reference."""

    reference: str
    reference_source: ReferenceSource


M = TypeVar("M", bound=BaseModel)


def read_records(path: Path | str) -> list[ClauseRecord]:
    """Load records from a JSON Lines file."""

    return read_jsonl(path, ClauseRecord)


def iter_records(path: Path | str) -> Iterator[ClauseRecord]:
    return iter_jsonl(path, ClauseRecord)


def read_jsonl(path: Path | str, model: type[M]) -> list[M]:
    return list(iter_jsonl(path, model))


def iter_jsonl(path: Path | str, model: type[M]) -> Iterator[M]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield model.model_validate_json(line)


def write_records(records: Iterable[BaseModel], path: Path | str) -> Path:
    """Write records to a JSON Lines file, creating parent directories."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")
    return path
