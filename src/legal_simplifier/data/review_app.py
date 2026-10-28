"""``lcs review``: a small Gradio page for checking golden-set references.

For each clause it shows the original, the draft reference (editable), the
required entities (green = in the current text, red = missing) and the
reading grade before and after. Every decision is appended to the golden
set's review log with the reviewer's name and a timestamp
(:func:`legal_simplifier.data.golden.record_review`).

The page is a thin layer over plain functions below, which the tests call
directly.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from legal_simplifier.data.entities import check_entity_presence
from legal_simplifier.data.golden import GoldenItem, golden_status, load_golden, record_review


def reading_grade(text: str) -> float | None:
    if not text.strip():
        return None
    import textstat

    return round(float(textstat.flesch_kincaid_grade(text)), 1)  # pyright: ignore[reportAttributeAccessIssue]


def entities_html(entities: Sequence[str], text: str) -> str:
    """Entity chips: green when ``text`` keeps the entity, red when it is missing."""

    if not entities:
        return "<p><em>No required entities detected for this clause.</em></p>"
    presence = check_entity_presence(entities, text)
    chips = []
    for entity in entities:
        kept = presence.get(entity, False)
        colour = "#1a7f37" if kept else "#cf222e"
        label = "kept" if kept else "missing"
        chips.append(
            f'<span style="display:inline-block;margin:2px;padding:2px 8px;border-radius:12px;'
            f'border:1px solid {colour};color:{colour}" title="{label}">'
            f"{'✓' if kept else '✗'} {html.escape(entity)}</span>"
        )
    return "".join(chips)


def queue_order(items: Sequence[GoldenItem]) -> list[GoldenItem]:
    """Unreviewed items first (in file order), then everything already reviewed."""

    return [i for i in items if i.review_status == "unreviewed"] + [
        i for i in items if i.review_status != "unreviewed"
    ]


def next_item_id(items: Sequence[GoldenItem], current_id: str | None) -> str | None:
    """The next unreviewed item after ``current_id`` (wrapping), or ``None`` when all are done."""

    ids = [i.id for i in items]
    start = ids.index(current_id) + 1 if current_id in ids else 0
    for offset in range(len(ids)):
        item = items[(start + offset) % len(ids)]
        if item.review_status == "unreviewed":
            return item.id  # the current item comes last, so a skip moves on when it can
    return None


def submit(
    golden_dir: Path | str,
    *,
    reviewer: str,
    item_id: str,
    text: str,
    note: str | None = None,
    reject: bool = False,
) -> GoldenItem:
    """Record a decision: reject, approve (text unchanged) or edit (text changed)."""

    items = {i.id: i for i in load_golden(golden_dir)}
    if item_id not in items:
        raise KeyError(item_id)
    if reject:
        return record_review(golden_dir, item_id, reviewer=reviewer, action="reject", note=note)
    unchanged = " ".join(text.split()) == " ".join(items[item_id].reference.split())
    return record_review(
        golden_dir,
        item_id,
        reviewer=reviewer,
        action="approve" if unchanged else "edit",
        reference_after=None if unchanged else text,
        note=note,
    )


def view(golden_dir: Path | str, item_id: str | None) -> dict[str, Any]:
    """Everything the page shows for one item (or a "done" view)."""

    items = load_golden(golden_dir)
    status = golden_status(items)
    done = status["n_items"] - status["counts"]["unreviewed"]
    progress = f"**{done} / {status['n_items']} reviewed** (status: {status['status']})"
    by_id = {i.id: i for i in items}
    if item_id is None or item_id not in by_id:
        item_id = next_item_id(items, None)
    if item_id is None:
        return {"id": None, "progress": progress + " — all items reviewed."}
    item = by_id[item_id]
    return {
        "id": item.id,
        "progress": progress,
        "header": (
            f"`{item.id}` · {item.category} · draft from **{item.draft_source}** · "
            f"current status **{item.review_status}**"
        ),
        "clause": item.clause,
        "reference": item.reference,
        "entities_html": entities_html(item.entities, item.reference),
        "grades": f"Reading grade (FKGL): clause {reading_grade(item.clause)} → reference {reading_grade(item.reference)}",
    }


def build_app(golden_dir: Path | str, reviewer: str) -> Any:
    """Build (but do not launch) the Gradio Blocks app."""

    import gradio as gr

    golden_dir = Path(golden_dir)
    if not reviewer.strip():
        raise ValueError("a reviewer name is required")

    def show(item_id: str | None):
        v = view(golden_dir, item_id)
        if v["id"] is None:
            return (None, v["progress"], "", "", "", "", "")
        return (
            v["id"],
            v["progress"],
            v["header"],
            v["clause"],
            v["reference"],
            v["entities_html"],
            v["grades"],
        )

    def decide(item_id: str | None, text: str, note: str, reject: bool):
        if item_id:
            submit(
                golden_dir, reviewer=reviewer, item_id=item_id, text=text, note=note, reject=reject
            )
        items = load_golden(golden_dir)
        return (*show(next_item_id(items, item_id)), "")

    def skip(item_id: str | None):
        return show(next_item_id(load_golden(golden_dir), item_id))

    with gr.Blocks(title="Golden set review") as app:
        gr.Markdown(f"## Golden set review\nReviewer: **{html.escape(reviewer)}**")
        current = gr.State(None)
        progress = gr.Markdown()
        header = gr.Markdown()
        clause = gr.Textbox(label="Original clause", lines=6, interactive=False)
        reference = gr.Textbox(label="Reference (edit to correct it)", lines=6)
        entity_view = gr.HTML(label="Required entities")
        grades = gr.Markdown()
        note = gr.Textbox(label="Note (optional)", lines=1)
        with gr.Row():
            accept = gr.Button("Accept (approve, or save my edit)", variant="primary")
            reject = gr.Button("Reject clause", variant="stop")
            skip_button = gr.Button("Skip")
        outputs = [current, progress, header, clause, reference, entity_view, grades]

        reference.change(
            lambda item_id, text: entities_html(
                next((i.entities for i in load_golden(golden_dir) if i.id == item_id), []), text
            ),
            inputs=[current, reference],
            outputs=entity_view,
        )
        accept.click(
            lambda i, t, n: decide(i, t, n, False), [current, reference, note], [*outputs, note]
        )
        reject.click(
            lambda i, t, n: decide(i, t, n, True), [current, reference, note], [*outputs, note]
        )
        skip_button.click(skip, [current], outputs)
        app.load(lambda: show(None), None, outputs)
    return app
