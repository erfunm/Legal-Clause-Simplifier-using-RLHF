"""``lcs label-judge``: a local Gradio page for scoring outputs on the judge's rubric.

Built from the golden-set review tool's pieces (entity chips, reading grade,
append-only log) so the two pages look and behave alike. The system that wrote
an output is never shown. The logic is plain functions below, tested directly.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from legal_simplifier.data.review_app import entities_html, reading_grade
from legal_simplifier.eval.judge import CRITERIA
from legal_simplifier.eval.labels import (
    label_progress,
    latest_labels,
    load_queue,
    next_unlabelled,
    record_label,
)

GUIDE = {
    "meaning_preserved": "5 same legal meaning · 3 main point right, a detail lost · 1 different or wrong meaning",
    "no_added_facts": "5 nothing added · 3 a minor addition or assumption · 1 invented obligations, amounts, parties or advice",
    "plain_language": "5 short, everyday, about grade 8 · 3 understandable but legalistic · 1 as hard as the clause, or garbled",
    "obligations_intact": "5 every must / may / must not, condition, exception and time limit kept · 3 one weakened or dropped · 1 several lost or flipped",
}


def view(run_dir: Path | str, item_id: str | None, reviewer: str) -> dict[str, Any]:
    queue = load_queue(run_dir)
    done = {lab["queue_id"] for lab in latest_labels(run_dir) if lab["reviewer"] == reviewer}
    progress = label_progress(run_dir, reviewer)
    text = f"**{progress['n_labelled']} / {progress['n_items']} labelled** by {reviewer}"
    by_id = {q["queue_id"]: q for q in queue}
    if item_id is None or item_id not in by_id:
        item_id = next_unlabelled(queue, done, None)
    if item_id is None:
        return {"id": None, "progress": text + " — all done."}
    entry = by_id[item_id]
    return {
        "id": item_id,
        "progress": text,
        "header": f"`{item_id}` · {entry.get('category')}",  # no system name: labels are blind
        "clause": entry["clause"],
        "output": entry["output"],
        "entities_html": entities_html(entry.get("entities", []), entry["output"]),
        "grades": f"Reading grade (FKGL): clause {reading_grade(entry['clause'])} → output {reading_grade(entry['output'])}",
    }


def submit(
    run_dir: Path | str,
    *,
    reviewer: str,
    item_id: str,
    scores: dict[str, int],
    note: str | None = None,
) -> str | None:
    """Record the scores and return the next unlabelled item for this reviewer."""

    record_label(run_dir, item_id, reviewer=reviewer, scores=scores, note=note)
    done = {lab["queue_id"] for lab in latest_labels(run_dir) if lab["reviewer"] == reviewer}
    return next_unlabelled(load_queue(run_dir), done, item_id)


def build_app(run_dir: Path | str, reviewer: str) -> Any:
    import gradio as gr

    run_dir = Path(run_dir)
    if not reviewer.strip():
        raise ValueError("a reviewer name is required")

    def show(item_id: str | None):
        v = view(run_dir, item_id, reviewer)
        if v["id"] is None:
            return (None, v["progress"], "", "", "", "", "")
        return (
            v["id"],
            v["progress"],
            v["header"],
            v["clause"],
            v["output"],
            v["entities_html"],
            v["grades"],
        )

    with gr.Blocks(title="Judge validation labels") as app:
        gr.Markdown(
            f"## Score outputs on the judge's rubric\nReviewer: **{html.escape(reviewer)}**"
        )
        current = gr.State(None)
        progress = gr.Markdown()
        header = gr.Markdown()
        clause = gr.Textbox(label="Original clause", lines=6, interactive=False)
        output = gr.Textbox(label="Rewrite to score", lines=6, interactive=False)
        entity_view = gr.HTML(label="Required entities")
        grades = gr.Markdown()
        radios = [
            gr.Radio(
                choices=[1, 2, 3, 4, 5], label=criterion.replace("_", " "), info=GUIDE[criterion]
            )
            for criterion in CRITERIA
        ]
        note = gr.Textbox(label="Note (optional)", lines=1)
        with gr.Row():
            save = gr.Button("Save and next", variant="primary")
            skip = gr.Button("Skip")
        outputs = [current, progress, header, clause, output, entity_view, grades]

        def decide(item_id: str | None, note_text: str, *values: int | None):
            if item_id and all(v is not None for v in values):
                nxt = submit(
                    run_dir,
                    reviewer=reviewer,
                    item_id=item_id,
                    scores={c: int(v) for c, v in zip(CRITERIA, values, strict=True)},  # type: ignore[arg-type]
                    note=note_text,
                )
                return (*show(nxt), "", *([None] * len(CRITERIA)))
            return (*show(item_id), note_text, *values)  # incomplete: stay on the item

        def next_item(item_id: str | None):
            queue = load_queue(run_dir)
            done = {
                lab["queue_id"] for lab in latest_labels(run_dir) if lab["reviewer"] == reviewer
            }
            return show(next_unlabelled(queue, done, item_id))

        save.click(decide, [current, note, *radios], [*outputs, note, *radios])
        skip.click(next_item, [current], outputs)
        app.load(lambda: show(None), None, outputs)
    return app
