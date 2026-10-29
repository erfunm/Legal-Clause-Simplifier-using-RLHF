"""Run the reward-hacking detectors over saved outputs and summarise what they find.

Input: JSON Lines where each row has a ``clause`` and either one output
(``output``, ``prediction`` or ``completion``) or several (``outputs``, as in the
``samples_<split>.jsonl`` files from ``lcs build-preferences``). ``entities``
are used when present and extracted from the clause otherwise. An optional
``system`` field groups the counts (for example SFT vs GRPO outputs).

Output: ``summary.json`` (rows scanned and, per detector, the count and rate,
overall and per system) and ``flagged.jsonl`` (every flagged output with the
detectors that fired and why), for picking before/after examples.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from legal_simplifier.data.entities import extract_entities
from legal_simplifier.rewards.hacking import DETECTORS, run_detectors

OUTPUT_FIELDS = ("output", "prediction", "completion")


@dataclass
class ScanSummary:
    n_outputs: int = 0
    flagged: dict[str, int] = field(default_factory=lambda: dict.fromkeys(DETECTORS, 0))
    n_any_flagged: int = 0
    by_system: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        n = max(1, self.n_outputs)
        return {
            "n_outputs": self.n_outputs,
            "n_any_flagged": self.n_any_flagged,
            "detectors": {
                name: {"count": count, "rate": round(count / n, 4)}
                for name, count in self.flagged.items()
            },
            "by_system": self.by_system,
        }


def iter_outputs(rows: Iterable[Mapping[str, Any]]) -> Iterator[dict[str, Any]]:
    """One item per (clause, output), whatever the row layout."""

    for index, row in enumerate(rows):
        if "clause" not in row:
            raise ValueError(f"row {index} has no 'clause'")
        if "outputs" in row:
            outputs = list(row["outputs"])
        else:
            field_name = next((f for f in OUTPUT_FIELDS if f in row), None)
            if field_name is None:
                raise ValueError(f"row {index} has none of {OUTPUT_FIELDS} or 'outputs'")
            outputs = [row[field_name]]
        entities = row.get("entities")
        if entities is None:
            entities = extract_entities(row["clause"])
        for sample, output in enumerate(outputs):
            yield {
                "id": row.get("id", str(index)),
                "sample": sample,
                "system": row.get("system", "all"),
                "clause": row["clause"],
                "output": output or "",
                "entities": list(entities),
            }


def scan(rows: Iterable[Mapping[str, Any]]) -> tuple[ScanSummary, list[dict[str, Any]]]:
    summary = ScanSummary()
    flagged_rows: list[dict[str, Any]] = []
    for item in iter_outputs(rows):
        findings = run_detectors(item["clause"], item["output"], item["entities"])
        fired = {name: f.detail for name, f in findings.items() if f.flagged}
        summary.n_outputs += 1
        system = summary.by_system.setdefault(
            item["system"], {"n_outputs": 0, **dict.fromkeys(DETECTORS, 0)}
        )
        system["n_outputs"] += 1
        for name in fired:
            summary.flagged[name] += 1
            system[name] += 1
        if fired:
            summary.n_any_flagged += 1
            flagged_rows.append({**item, "detectors": fired})
    return summary, flagged_rows


def scan_file(path: Path | str, out_dir: Path | str) -> ScanSummary:
    path, out_dir = Path(path), Path(out_dir)
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    summary, flagged = scan(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps({"source": str(path), **summary.to_dict()}, indent=2), encoding="utf-8"
    )
    with (out_dir / "flagged.jsonl").open("w", encoding="utf-8") as handle:
        for row in flagged:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return summary
