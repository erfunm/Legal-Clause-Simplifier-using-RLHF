"""Regenerate ``cuad_mini.json``: a tiny synthetic file in CUAD v1's SQuAD layout.

The clause text is written for the tests (not copied from CUAD). It mirrors
CUAD's structure exactly (one paragraph per contract holding the whole
contract, one question per category with the long "Highlight the parts..."
prompt, ids ``<title>__<Category>``, answers as character offsets) and covers
the extraction edge cases:

* overlapping and adjacent answer spans (merged),
* the same span labelled with two categories (kept once, both labels),
* spans too short and too long (filtered and counted),
* a category we do not select ("Document Name"),
* a clause repeated in another contract (removed as a near-duplicate).

Run: ``uv run python tests/fixtures/build_cuad_mini.py``
"""

from __future__ import annotations

import json
from pathlib import Path

QUESTION = (
    'Highlight the parts (if any) of this contract related to "{category}" that should be '
    "reviewed by a lawyer. Details: synthetic test question."
)

GOVERNING_LAW = (
    "This Agreement shall be governed by and construed in accordance with the laws of the "
    "State of Delaware, without regard to its conflict of laws principles."
)

CONTRACTS = [
    {
        "title": "ACMEWIDGETS_2019-EX-10.1-SUPPLY AGREEMENT",
        "sections": [
            ("Document Name", "SUPPLY AGREEMENT"),
            ("Governing Law", GOVERNING_LAW),
            # Two spans of one sentence pair, split by a space: merged into one clause.
            (
                "Termination For Convenience",
                "Either party may terminate this Agreement for any reason upon ninety (90) days prior written notice.",
            ),
            (
                "Termination For Convenience",
                "Upon such termination, Buyer shall pay for all Products delivered before the effective date of termination.",
            ),
            # Same text labelled with two categories: one candidate, both labels.
            (
                "Cap On Liability|Liquidated Damages",
                "In no event shall Acme Widgets, Inc. be liable for more than $250,000, and any late delivery shall incur liquidated damages of 2% of the order value per week.",
            ),
            # Too short (under 6 words).
            ("Anti-Assignment", "Neither party may assign."),
        ],
    },
    {
        "title": "GLOBEXCORP_2020-EX-10.4-DISTRIBUTION AGREEMENT",
        "sections": [
            # Near-duplicate of the governing-law clause above (one word differs).
            ("Governing Law", GOVERNING_LAW.replace("conflict of laws", "conflicts of law")),
            (
                "Exclusivity",
                'Globex Corporation (the "Distributor") shall be the exclusive distributor of the Products in the Territory commencing January 1, 2021.',
            ),
            (
                "Non-Compete",
                "During the Term and for twelve (12) months thereafter, the Distributor shall not sell any product that competes with the Products.",
            ),
            # Too long (over 1,500 characters).
            (
                "Post-Termination Services",
                " ".join(
                    f"Following expiration, the Distributor shall provide transition service number {i} to the Supplier at the rates then in effect."
                    for i in range(15)
                ),
            ),
        ],
    },
    {
        "title": "INITECH_2021-EX-10.2-LICENSE AGREEMENT",
        "sections": [
            (
                "Ip Ownership Assignment",
                "All right, title and interest in any Deliverables created by the Consultant shall vest in Initech LLC upon creation.",
            ),
            (
                "Change Of Control",
                "If Initech LLC undergoes a change of control, the Licensor may terminate this Agreement upon thirty (30) days written notice.",
            ),
            (
                "Renewal Term",
                "This Agreement shall automatically renew for successive one (1) year terms unless either party gives notice of non-renewal at least sixty (60) days before the end of the then-current term.",
            ),
        ],
    },
    {
        "title": "HOOLI_2022-EX-10.9-SERVICES AGREEMENT",
        "sections": [
            (
                "Governing Law",
                "This Agreement is governed by the laws of England and Wales and the parties submit to the courts of London.",
            ),
            (
                "Anti-Assignment",
                "Neither party may assign this Agreement without the prior written consent of the other party, which shall not be unreasonably withheld.",
            ),
            (
                "Cap On Liability",
                "Each party's total liability under this Agreement shall not exceed the fees paid in the twelve (12) months preceding the claim, or USD 1,000,000, whichever is lower.",
            ),
        ],
    },
]


def build() -> dict:
    data = []
    for contract in CONTRACTS:
        context_parts: list[str] = [f"{contract['title']}\n\n"]
        answers: dict[str, list[dict]] = {}
        for labels, text in contract["sections"]:
            start = sum(len(p) for p in context_parts)
            context_parts.append(text + " ")
            for label in labels.split("|"):
                answers.setdefault(label, []).append({"text": text, "answer_start": start})
            context_parts.append("\n\nSection break. ")
        context = "".join(context_parts)
        # Make the termination pair adjacent (separated by one space only).
        qas = []
        categories = sorted(set(answers) | {"Insurance"})  # Insurance: a question with no answers
        for category in categories:
            spans = answers.get(category, [])
            qas.append(
                {
                    "answers": spans,
                    "id": f"{contract['title']}__{category}",
                    "question": QUESTION.format(category=category),
                    "is_impossible": not spans,
                }
            )
        data.append({"title": contract["title"], "paragraphs": [{"context": context, "qas": qas}]})
    return {"version": "aok_v1.0", "data": data}


def _make_adjacent(cuad: dict) -> dict:
    """Rewrite the first contract so its two termination spans are separated by one space."""

    paragraph = cuad["data"][0]["paragraphs"][0]
    qa = next(q for q in paragraph["qas"] if q["id"].endswith("__Termination For Convenience"))
    first, second = qa["answers"]
    gap_start = first["answer_start"] + len(first["text"])
    context = paragraph["context"]
    removed = second["answer_start"] - gap_start - 1
    paragraph["context"] = context[:gap_start] + " " + context[second["answer_start"] :]
    for q in paragraph["qas"]:
        for answer in q["answers"]:
            if answer["answer_start"] > gap_start:
                answer["answer_start"] -= removed
    return cuad


if __name__ == "__main__":
    cuad = _make_adjacent(build())
    for document in cuad["data"]:
        ctx = document["paragraphs"][0]["context"]
        for qa in document["paragraphs"][0]["qas"]:
            for a in qa["answers"]:
                assert ctx[a["answer_start"] : a["answer_start"] + len(a["text"])] == a["text"]
    out = Path(__file__).with_name("cuad_mini.json")
    out.write_text(json.dumps(cuad, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")
