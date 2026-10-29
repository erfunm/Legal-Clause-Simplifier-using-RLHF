"""Human labels on the judge's rubric, for validating the judge (``lcs label-judge``).

Layout under ``results/<run>/human_labels/``:

* ``queue.jsonl``: the outputs to label, written once. ``n`` (default 80)
  outputs sampled evenly across systems and categories with a fixed seed.
  Each entry keeps its system for the agreement step, but the labelling page
  never shows it.
* ``labels.jsonl``: append-only, one line per decision: queue id, reviewer,
  UTC time, a 1-5 score per criterion and an optional note. A later label by
  the same reviewer for the same output replaces the earlier one in
  :func:`latest_labels`; nothing is overwritten on disk.

The page (:mod:`legal_simplifier.eval.label_app`) reuses the golden-set review
tool's entity chips and reading grade.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from legal_simplifier.eval.evaluate import load_run, read_jsonl
from legal_simplifier.eval.judge import CRITERIA

LABEL_DIR = "human_labels"
QUEUE_FILE = "queue.jsonl"
LABELS_FILE = "labels.jsonl"


def label_dir(run_dir: Path | str) -> Path:
    return Path(run_dir) / LABEL_DIR


def queue_id(row: Mapping[str, Any]) -> str:
    text = f"{row['dataset']}\x1f{row['id']}\x1f{row['system']}"
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def build_queue(run_dir: Path | str, *, n: int = 80, seed: int = 5) -> list[dict[str, Any]]:
    """Sample ``n`` non-empty outputs, round-robin over systems, in a shuffled order."""

    path = label_dir(run_dir) / QUEUE_FILE
    if path.exists():
        raise FileExistsError(f"{path} exists; delete it only if no labels depend on it")
    _, rows = load_run(run_dir)
    rng = random.Random(seed)
    by_system: dict[str, list[dict[str, Any]]] = {}
    for row in sorted(rows, key=lambda r: (r["system"], r["dataset"], r["id"])):
        if str(row["output"]).strip():
            by_system.setdefault(row["system"], []).append(row)
    for members in by_system.values():
        rng.shuffle(members)
    chosen: list[dict[str, Any]] = []
    while len(chosen) < n and any(by_system.values()):
        for system in sorted(by_system):
            if by_system[system] and len(chosen) < n:
                chosen.append(by_system[system].pop())
    rng.shuffle(chosen)
    queue = [
        {
            "queue_id": queue_id(r),
            "dataset": r["dataset"],
            "id": r["id"],
            "system": r["system"],
            "category": r.get("category"),
            "clause": r["clause"],
            "output": r["output"],
            "entities": r.get("entities", []),
        }
        for r in chosen
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in queue:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    (label_dir(run_dir) / LABELS_FILE).touch()
    return queue


def load_queue(run_dir: Path | str) -> list[dict[str, Any]]:
    return read_jsonl(label_dir(run_dir) / QUEUE_FILE)


def record_label(
    run_dir: Path | str,
    item_id: str,
    *,
    reviewer: str,
    scores: Mapping[str, int],
    note: str | None = None,
) -> dict[str, Any]:
    if not reviewer.strip():
        raise ValueError("a reviewer name is required")
    if item_id not in {q["queue_id"] for q in load_queue(run_dir)}:
        raise KeyError(item_id)
    if set(scores) != set(CRITERIA) or not all(
        isinstance(v, int) and 1 <= v <= 5 for v in scores.values()
    ):
        raise ValueError(f"need a 1-5 score for each of {CRITERIA}")
    event = {
        "queue_id": item_id,
        "reviewer": reviewer.strip(),
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "scores": dict(scores),
        "note": note or None,
    }
    with (label_dir(run_dir) / LABELS_FILE).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    return event


def latest_labels(run_dir: Path | str) -> list[dict[str, Any]]:
    """The last label per (output, reviewer), joined with the queue entry."""

    queue = {q["queue_id"]: q for q in load_queue(run_dir)}
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for event in read_jsonl(label_dir(run_dir) / LABELS_FILE):
        latest[(event["queue_id"], event["reviewer"])] = event
    return [{**queue[qid], **event} for (qid, _), event in latest.items() if qid in queue]


def label_progress(run_dir: Path | str, reviewer: str | None = None) -> dict[str, int]:
    queue = load_queue(run_dir)
    done = {
        lab["queue_id"]
        for lab in latest_labels(run_dir)
        if reviewer is None or lab["reviewer"] == reviewer
    }
    return {"n_items": len(queue), "n_labelled": len(done & {q["queue_id"] for q in queue})}


def next_unlabelled(
    queue: Sequence[Mapping[str, Any]], done: set[str], current: str | None
) -> str | None:
    ids = [q["queue_id"] for q in queue]
    start = ids.index(current) + 1 if current in ids else 0
    for offset in range(len(ids)):
        candidate = ids[(start + offset) % len(ids)]
        if candidate not in done:
            return candidate
    return None
