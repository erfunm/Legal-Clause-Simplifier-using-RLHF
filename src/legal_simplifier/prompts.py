"""Prompts for the model being trained (the "student").

Prompts are chat messages; the tokenizer's chat template turns them into text.
The same builder is used by SFT, GRPO, DPO, evaluation and serving so that
training and inference see identical inputs.
"""

from __future__ import annotations

from typing import TypedDict

SYSTEM_PROMPT = (
    "You rewrite legal contract clauses in plain English for a non-lawyer. "
    "Use short sentences and everyday words, aiming for a grade 8 reading level. "
    "Keep every party, date, amount, duration, percentage and defined term, and keep "
    "every obligation (must), permission (may) and prohibition (must not). "
    "Do not add facts, advice or commentary. Reply with the rewritten clause only."
)


class Message(TypedDict):
    role: str
    content: str


def normalise_clause(clause: str) -> str:
    """Collapse the whitespace and line breaks that PDF-extracted clauses carry."""

    return " ".join(clause.split())


def build_messages(clause: str, category: str | None = None) -> list[Message]:
    """Return the system + user messages asking for a simplification of ``clause``."""

    clause = normalise_clause(clause)
    if not clause:
        raise ValueError("clause is empty")
    header = f"Clause type: {category}\n\n" if category else ""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{header}Clause:\n{clause}"},
    ]


def build_completion(reference: str) -> list[Message]:
    """Return the assistant message holding a target simplification."""

    return [{"role": "assistant", "content": reference.strip()}]
