"""The golden test set: ~150 test-split clauses whose references a human checks.

Layout under ``data/golden/``:

* ``drafts.jsonl``: the selected test records with their silver (Claude or
  mock) references. Written once by ``lcs build-golden``; never edited.
* ``reviews.jsonl``: an append-only log of review events (who, when, what
  action, the text before and after). Nothing is overwritten, so the history
  of every reference is kept.
* ``golden.jsonl`` and ``status.json``: drafts with the latest review applied,
  re-materialised after every review.

Until a person reviews an item it stays ``unreviewed`` with its silver
reference, and results computed on the set must say so.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from legal_simplifier.data.schema import ClauseRecord, ReferenceSource, read_jsonl, write_records

ReviewStatus = Literal["unreviewed", "approved", "edited", "rejected"]
ReviewAction = Literal["approve", "edit", "reject"]

DRAFTS_FILE = "drafts.jsonl"
REVIEWS_FILE = "reviews.jsonl"
GOLDEN_FILE = "golden.jsonl"
STATUS_FILE = "status.json"


class GoldenItem(ClauseRecord):
    draft_reference: str
    draft_source: ReferenceSource
    review_status: ReviewStatus = "unreviewed"
    reviewer: str | None = None
    reviewed_at: str | None = None
    review_note: str | None = None


class ReviewEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str
    reviewer: str
    action: ReviewAction
    timestamp: str
    reference_before: str
    reference_after: str
    note: str | None = None


def select_golden(
    test_records: Sequence[ClauseRecord], *, size: int = 150, seed: int = 7
) -> list[ClauseRecord]:
    """Pick ``size`` test records, as evenly across categories as availability allows.

    Each category gets an equal share; categories with too few clauses give
    their unused share to the others, round-robin.
    """

    if any(r.split not in (None, "test") for r in test_records):
        raise ValueError("the golden set must come from the test split only")
    rng = random.Random(seed)
    by_category: dict[str, list[ClauseRecord]] = {}
    for record in sorted(test_records, key=lambda r: r.id):
        by_category.setdefault(record.category, []).append(record)
    for records in by_category.values():
        rng.shuffle(records)

    size = min(size, len(test_records))
    chosen: dict[str, int] = dict.fromkeys(by_category, 0)
    categories = sorted(by_category)
    while sum(chosen.values()) < size:
        progressed = False
        for category in categories:
            if sum(chosen.values()) >= size:
                break
            if chosen[category] < len(by_category[category]):
                chosen[category] += 1
                progressed = True
        if not progressed:
            break
    selected = [r for c in categories for r in by_category[c][: chosen[c]]]
    return sorted(selected, key=lambda r: (r.category, r.id))


def build_golden(
    test_records: Sequence[ClauseRecord], golden_dir: Path | str, *, size: int = 150, seed: int = 7
) -> list[GoldenItem]:
    """Write ``drafts.jsonl`` for a new golden set. Refuses to overwrite one with reviews."""

    golden_dir = Path(golden_dir)
    if (golden_dir / REVIEWS_FILE).exists() and (golden_dir / REVIEWS_FILE).stat().st_size:
        raise FileExistsError(
            f"{golden_dir} already has reviews; move it aside before building a new golden set"
        )
    drafts = [
        GoldenItem(
            **record.model_dump(),
            draft_reference=record.reference,
            draft_source=record.reference_source,
        )
        for record in select_golden(test_records, size=size, seed=seed)
    ]
    write_records(drafts, golden_dir / DRAFTS_FILE)
    (golden_dir / REVIEWS_FILE).touch()
    materialise(golden_dir)
    return drafts


def read_reviews(golden_dir: Path | str) -> list[ReviewEvent]:
    path = Path(golden_dir) / REVIEWS_FILE
    return read_jsonl(path, ReviewEvent) if path.exists() else []


def apply_reviews(drafts: Sequence[GoldenItem], reviews: Sequence[ReviewEvent]) -> list[GoldenItem]:
    """Apply the latest review event per item to the drafts."""

    latest: dict[str, ReviewEvent] = {}
    for event in reviews:
        latest[event.item_id] = event
    status_for: dict[ReviewAction, ReviewStatus] = {
        "approve": "approved",
        "edit": "edited",
        "reject": "rejected",
    }
    items = []
    for draft in drafts:
        event = latest.get(draft.id)
        if event is None:
            items.append(draft)
            continue
        update: dict[str, Any] = {
            "review_status": status_for[event.action],
            "reviewer": event.reviewer,
            "reviewed_at": event.timestamp,
            "review_note": event.note,
        }
        if event.action in ("approve", "edit"):
            update["reference"] = event.reference_after
            update["reference_source"] = "human"
        items.append(draft.model_copy(update=update))
    return items


def load_golden(golden_dir: Path | str) -> list[GoldenItem]:
    golden_dir = Path(golden_dir)
    drafts = read_jsonl(golden_dir / DRAFTS_FILE, GoldenItem)
    return apply_reviews(drafts, read_reviews(golden_dir))


def golden_status(items: Sequence[GoldenItem]) -> dict[str, Any]:
    counts = Counter(item.review_status for item in items)
    reviewed = counts["approved"] + counts["edited"] + counts["rejected"]
    overall = (
        "unreviewed"
        if reviewed == 0
        else "reviewed"
        if reviewed == len(items)
        else "partially_reviewed"
    )
    return {
        "status": overall,
        "n_items": len(items),
        "counts": {s: counts.get(s, 0) for s in ("unreviewed", "approved", "edited", "rejected")},
        "reviewers": sorted({i.reviewer for i in items if i.reviewer}),
        "draft_sources": dict(Counter(i.draft_source for i in items)),
    }


def materialise(golden_dir: Path | str) -> list[GoldenItem]:
    """Rewrite ``golden.jsonl`` and ``status.json`` from drafts + review log."""

    golden_dir = Path(golden_dir)
    items = load_golden(golden_dir)
    write_records(items, golden_dir / GOLDEN_FILE)
    (golden_dir / STATUS_FILE).write_text(json.dumps(golden_status(items), indent=2) + "\n")
    return items


def record_review(
    golden_dir: Path | str,
    item_id: str,
    *,
    reviewer: str,
    action: ReviewAction,
    reference_after: str | None = None,
    note: str | None = None,
) -> GoldenItem:
    """Append one review event and return the updated item."""

    reviewer = reviewer.strip()
    if not reviewer:
        raise ValueError("a reviewer name is required")
    golden_dir = Path(golden_dir)
    current = {item.id: item for item in load_golden(golden_dir)}
    if item_id not in current:
        raise KeyError(f"no golden item {item_id!r}")
    before = current[item_id].reference
    if action == "edit":
        if reference_after is None or not reference_after.strip():
            raise ValueError("an edit needs the corrected reference text")
        after = " ".join(reference_after.split())
    else:
        after = before
    event = ReviewEvent(
        item_id=item_id,
        reviewer=reviewer,
        action=action,
        timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
        reference_before=before,
        reference_after=after,
        note=note.strip() if note and note.strip() else None,
    )
    with (golden_dir / REVIEWS_FILE).open("a", encoding="utf-8") as handle:
        handle.write(event.model_dump_json() + "\n")
    items = materialise(golden_dir)
    return next(item for item in items if item.id == item_id)
