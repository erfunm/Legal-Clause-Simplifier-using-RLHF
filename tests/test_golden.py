"""Golden set selection, append-only review log, and the review page's logic."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from legal_simplifier.data.golden import (
    REVIEWS_FILE,
    build_golden,
    golden_status,
    load_golden,
    read_reviews,
    record_review,
    select_golden,
)
from legal_simplifier.data.review_app import build_app, entities_html, next_item_id, submit, view
from legal_simplifier.data.schema import ClauseRecord

from .conftest import make_record


def _test_records(per_category: dict[str, int]) -> list[ClauseRecord]:
    records = []
    for category, n in per_category.items():
        for i in range(n):
            record = make_record(
                f"{category[:3]}-{i}",
                f"k{i}",
                f"Clause {i} about {category}: Acme Inc. pays $100.",
                category=category,
                reference=f"Acme Inc. pays $100 ({i}).",
                reference_source="claude",
            )
            records.append(
                record.model_copy(update={"split": "test", "entities": ["Acme Inc.", "$100"]})
            )
    return records


@pytest.fixture
def golden_dir(tmp_path: Path) -> Path:
    build_golden(_test_records({"Governing Law": 3, "Non-Compete": 3}), tmp_path, size=4)
    return tmp_path


def test_selection_is_balanced_and_redistributes_scarce_categories() -> None:
    records = _test_records({"A cat": 2, "B cat": 20, "C cat": 20})
    chosen = select_golden(records, size=12, seed=1)
    counts = Counter(r.category for r in chosen)
    assert counts == {"A cat": 2, "B cat": 5, "C cat": 5}
    assert select_golden(records, size=12, seed=1) == chosen
    assert len(select_golden(records, size=500)) == len(records)


def test_selection_refuses_non_test_records() -> None:
    record = _test_records({"A": 1})[0].model_copy(update={"split": "train"})
    with pytest.raises(ValueError, match="test split"):
        select_golden([record])


def test_new_golden_set_is_unreviewed_and_keeps_drafts(golden_dir: Path) -> None:
    items = load_golden(golden_dir)
    assert len(items) == 4
    assert {i.review_status for i in items} == {"unreviewed"}
    assert all(i.draft_reference == i.reference and i.draft_source == "claude" for i in items)
    status = json.loads((golden_dir / "status.json").read_text())
    assert status["status"] == "unreviewed"
    assert status["counts"]["unreviewed"] == 4


def test_reviews_are_appended_with_reviewer_and_time(golden_dir: Path) -> None:
    first, second, third = (i.id for i in load_golden(golden_dir)[:3])
    record_review(golden_dir, first, reviewer="Reviewer A", action="approve")
    edited = record_review(
        golden_dir,
        second,
        reviewer="Reviewer A",
        action="edit",
        reference_after="Acme Inc. must pay $100.",
    )
    record_review(golden_dir, third, reviewer="Reviewer B", action="reject", note="fragment")

    assert edited.reference == "Acme Inc. must pay $100."
    assert edited.reference_source == "human" and edited.review_status == "edited"
    by_id = {i.id: i for i in load_golden(golden_dir)}
    assert by_id[first].review_status == "approved" and by_id[first].reference_source == "human"
    assert by_id[third].review_status == "rejected" and by_id[third].review_note == "fragment"
    assert by_id[third].reference_source == "claude"  # rejecting does not make it human-verified

    events = read_reviews(golden_dir)
    assert [(e.item_id, e.reviewer, e.action) for e in events] == [
        (first, "Reviewer A", "approve"),
        (second, "Reviewer A", "edit"),
        (third, "Reviewer B", "reject"),
    ]
    assert all(e.timestamp for e in events)
    assert events[1].reference_before != events[1].reference_after

    status = golden_status(list(by_id.values()))
    assert status["status"] == "partially_reviewed"
    assert status["reviewers"] == ["Reviewer A", "Reviewer B"]


def test_a_later_review_wins_but_history_is_kept(golden_dir: Path) -> None:
    item = load_golden(golden_dir)[0].id
    record_review(golden_dir, item, reviewer="R", action="edit", reference_after="First fix.")
    record_review(golden_dir, item, reviewer="R", action="edit", reference_after="Second fix.")
    assert next(i for i in load_golden(golden_dir) if i.id == item).reference == "Second fix."
    assert [e.reference_after for e in read_reviews(golden_dir)] == ["First fix.", "Second fix."]


def test_review_validation(golden_dir: Path) -> None:
    item = load_golden(golden_dir)[0].id
    with pytest.raises(ValueError, match="reviewer"):
        record_review(golden_dir, item, reviewer="  ", action="approve")
    with pytest.raises(ValueError, match="corrected"):
        record_review(golden_dir, item, reviewer="R", action="edit", reference_after=" ")
    with pytest.raises(KeyError):
        record_review(golden_dir, "missing", reviewer="R", action="approve")


def test_rebuilding_a_reviewed_golden_set_is_refused(golden_dir: Path) -> None:
    record_review(golden_dir, load_golden(golden_dir)[0].id, reviewer="R", action="approve")
    with pytest.raises(FileExistsError):
        build_golden(_test_records({"Governing Law": 3}), golden_dir)
    assert (golden_dir / REVIEWS_FILE).read_text().strip()


def test_fully_reviewed_status(golden_dir: Path) -> None:
    for item in load_golden(golden_dir):
        record_review(golden_dir, item.id, reviewer="R", action="approve")
    assert json.loads((golden_dir / "status.json").read_text())["status"] == "reviewed"


# --- review page logic ----------------------------------------------------------


def test_entity_chips_show_kept_and_missing() -> None:
    page = entities_html(["Acme Inc.", "$100", "<b>"], "Acme Inc. pays.")
    assert "✓ Acme Inc." in page and "✗ $100" in page
    assert "&lt;b&gt;" in page and "<b>" not in page
    assert "No required entities" in entities_html([], "x")


def test_submit_detects_approve_versus_edit(golden_dir: Path) -> None:
    a, b = (i for i in load_golden(golden_dir)[:2])
    submit(golden_dir, reviewer="R", item_id=a.id, text="  " + a.reference + " ")
    submit(golden_dir, reviewer="R", item_id=b.id, text="Acme Inc. must pay $100.")
    actions = [e.action for e in read_reviews(golden_dir)]
    assert actions == ["approve", "edit"]
    submit(golden_dir, reviewer="R", item_id=a.id, text="ignored", reject=True)
    assert read_reviews(golden_dir)[-1].action == "reject"


def test_queue_moves_through_unreviewed_items(golden_dir: Path) -> None:
    items = load_golden(golden_dir)
    first = next_item_id(items, None)
    assert first == items[0].id
    assert next_item_id(items, first) == items[1].id
    assert next_item_id(items, items[-1].id) == items[0].id  # wraps around
    for item in items[:-1]:
        record_review(golden_dir, item.id, reviewer="R", action="approve")
    remaining = load_golden(golden_dir)
    assert next_item_id(remaining, items[-1].id) == items[-1].id  # skipping the last one stays put
    record_review(golden_dir, items[-1].id, reviewer="R", action="approve")
    assert next_item_id(load_golden(golden_dir), None) is None


def test_view_shows_progress_entities_and_grades(golden_dir: Path) -> None:
    v = view(golden_dir, None)
    assert v["progress"].startswith("**0 / 4 reviewed**")
    assert "✓ Acme Inc." in v["entities_html"]
    assert "Reading grade" in v["grades"]
    for item in load_golden(golden_dir):
        record_review(golden_dir, item.id, reviewer="R", action="approve")
    assert view(golden_dir, None)["id"] is None


def test_review_app_builds(golden_dir: Path) -> None:
    gr = pytest.importorskip("gradio")
    assert isinstance(build_app(golden_dir, "R"), gr.Blocks)
    with pytest.raises(ValueError):
        build_app(golden_dir, " ")
