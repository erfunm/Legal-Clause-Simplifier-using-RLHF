"""The Gradio demo behind ``space/app.py`` (a Hugging Face Space, not pushed by this repo).

Shows the clause and its simplification side by side, with the clause's entities
highlighted (green = kept in the rewrite, red = missing), the detector warnings,
the reading grade before and after, example clauses, and a "not legal advice"
banner. A backend is either the local model (:class:`LocalBackend`) or a running
``lcs serve`` (:class:`RemoteBackend`); both return the API's response shape.
"""

from __future__ import annotations

import html
import re
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from legal_simplifier.config import ServeConfig
from legal_simplifier.data.review_app import entities_html
from legal_simplifier.prompts import build_messages
from legal_simplifier.serving.analysis import DISCLAIMER

# Synthetic clauses written for the demo (not CUAD text), each with a condition or exception.
EXAMPLES: list[tuple[str, str]] = [
    (
        "The Licensee may terminate this Agreement on thirty (30) days' written notice, unless the "
        "Licensor has cured the breach within ten (10) days.",
        "Termination For Convenience",
    ),
    (
        "In no event shall Acme Widgets, Inc.'s aggregate liability exceed $250,000, except for "
        "liability arising from gross negligence.",
        "Cap On Liability",
    ),
    (
        "This Agreement shall automatically renew for successive one (1) year terms unless either "
        "party gives notice of non-renewal at least sixty (60) days before the end of the then-current term.",
        "Renewal Term",
    ),
    (
        "Each party shall keep the Confidential Information confidential and shall not disclose it to "
        "any third party, except as required by law.",
        "Confidentiality",
    ),
]

BANNER = f"⚠️ **{DISCLAIMER}**"


class Backend(Protocol):
    label: str

    def simplify(self, clause: str, category: str | None) -> dict[str, Any]: ...


class LocalBackend:
    """Runs the model in this process with the same engine and analysis as the API."""

    def __init__(self, config: ServeConfig, engine: Any | None = None):
        from legal_simplifier.serving.engine import make_engine

        self.config = config
        self.engine = engine or make_engine(config)
        self.label = f"local {config.model_label} ({self.engine.name})"

    def simplify(self, clause: str, category: str | None) -> dict[str, Any]:
        import time

        from legal_simplifier.serving.app import response_body

        clause = " ".join(clause.split())
        start = time.perf_counter()
        (generation,) = self.engine.generate(
            [build_messages(clause, category)], self.config.max_new_tokens
        )
        return response_body(
            "local",
            clause,
            generation,
            model=self.config.model_label,
            engine=self.engine.name,
            latency_ms=(time.perf_counter() - start) * 1000,
        )


class RemoteBackend:
    """Calls ``POST /simplify`` on a running ``lcs serve``."""

    def __init__(self, url: str, *, client: Any | None = None, timeout_s: float = 120.0):
        import httpx

        self.url = url.rstrip("/")
        self.client = client or httpx.Client(timeout=timeout_s)
        self.label = f"API at {self.url}"

    def simplify(self, clause: str, category: str | None) -> dict[str, Any]:
        response = self.client.post(
            f"{self.url}/simplify", json={"clause": clause, "category": category}
        )
        if response.status_code >= 400:
            detail = (
                response.json().get("detail")
                if "json" in response.headers.get("content-type", "")
                else response.text
            )
            raise ValueError(f"the API returned {response.status_code}: {detail}")
        return response.json()


def highlight(text: str, entities: Sequence[Mapping[str, Any]], *, only_kept: bool = False) -> str:
    """``text`` as HTML with each entity's occurrences marked green (kept) or red (missing).

    One pass over the raw text with every entity in one pattern, longest first, so an
    entity inside a longer one ("Licensor" in "the Licensor") is not marked twice.
    """

    wanted = [e for e in entities if e["kept"] or not only_kept]
    if not wanted:
        return f'<div style="line-height:1.6">{html.escape(text)}</div>'
    kept = {e["text"].lower(): e["kept"] for e in wanted}
    pattern = re.compile(
        r"(?<!\w)("
        + "|".join(re.escape(e["text"]) for e in sorted(wanted, key=lambda e: -len(e["text"])))
        + r")(?!\w)",
        re.IGNORECASE,
    )
    pieces, pos = [], 0
    for match in pattern.finditer(text):
        colour = "#1a7f37" if kept.get(match.group(0).lower(), True) else "#cf222e"
        pieces.append(html.escape(text[pos : match.start()]))
        pieces.append(
            f'<mark style="background:none;color:{colour};font-weight:600;'
            f'border-bottom:2px solid {colour}">{html.escape(match.group(0))}</mark>'
        )
        pos = match.end()
    pieces.append(html.escape(text[pos:]))
    return f'<div style="line-height:1.6">{"".join(pieces)}</div>'


def render(clause: str, result: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    """(original HTML, simplification HTML, entity chips HTML, warnings Markdown, grades Markdown)."""

    entities = result["entities"]
    simplification = result["simplification"]
    warnings = result["warnings"]
    warn_md = (
        "\n".join(
            f"- **{w['detector'].replace('_', ' ')}**: {w['message']}"
            + (f" ({w['detail']})" if w.get("detail") else "")
            for w in warnings
        )
        if warnings
        else "No warnings from the checks (they are heuristics: read the clause anyway)."
    )
    grades = result["reading_grade"]
    grade_md = f"Reading grade (FKGL): **{grades['clause']}** → **{grades['simplification']}**"
    return (
        highlight(clause, entities),
        highlight(simplification, entities, only_kept=True),
        entities_html([e["text"] for e in entities], simplification),
        warn_md,
        grade_md,
    )


def build_demo(backend: Backend) -> Any:
    import gradio as gr

    def run(clause: str, category: str) -> tuple[str, str, str, str, str]:
        if not clause.strip():
            return ("", "", "", "Paste a clause first.", "")
        try:
            result = backend.simplify(clause, category or None)
        except ValueError as exc:
            return ("", "", "", f"**Error:** {html.escape(str(exc))}", "")
        return render(clause, result)

    with gr.Blocks(title="Legal Clause Simplifier") as demo:
        gr.Markdown("# Legal Clause Simplifier\nPlain-English rewrites of contract clauses.")
        gr.Markdown(BANNER)
        with gr.Row():
            clause = gr.Textbox(label="Clause", lines=6, placeholder="Paste a contract clause…")
            category = gr.Textbox(
                label="Clause type (optional)", lines=1, placeholder="e.g. Governing Law"
            )
        go = gr.Button("Simplify", variant="primary")
        with gr.Row():
            original = gr.HTML(label="Original (entities: green = kept, red = missing)")
            simplified = gr.HTML(label="Plain English")
        chips = gr.HTML(label="Required entities")
        warnings = gr.Markdown()
        grades = gr.Markdown()
        gr.Examples(examples=[list(e) for e in EXAMPLES], inputs=[clause, category])
        gr.Markdown(f"Backend: {html.escape(backend.label)}")
        go.click(run, [clause, category], [original, simplified, chips, warnings, grades])
    return demo
