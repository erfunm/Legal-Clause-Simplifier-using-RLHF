"""What the API and the Space report next to a simplification.

Built from the same code as training and evaluation: the entity extractor and
matcher, the reward-hacking detectors (surfaced as warnings: a dropped
condition or a must/may flip is something a reader should double-check), and
the reading grade.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from legal_simplifier.data.entities import check_entity_presence, extract_entities
from legal_simplifier.rewards.components import fkgl
from legal_simplifier.rewards.hacking import run_detectors

DISCLAIMER = (
    "Not legal advice. This is an automated plain-English paraphrase and can lose or change "
    "meaning; the original clause governs."
)

WARNING_TEXT = {
    "condition_drop": "A condition, exception or time limit in the clause may be missing.",
    "modality_flip": "An obligation, permission or prohibition may have changed force (must / may / must not).",
    "truncation": "The rewrite is much shorter and may drop an obligation.",
    "verbatim_copy": "The rewrite mostly copies the clause.",
    "word_salad": "The rewrite strings the clause's names and numbers together with little else.",
}


@dataclass
class Analysis:
    entities: list[dict[str, Any]]
    warnings: list[dict[str, str]]
    reading_grade: dict[str, float | None]
    disclaimer: str = DISCLAIMER
    missing_entities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _grade(text: str) -> float | None:
    return round(fkgl(text), 1) if text.strip() else None


def analyse(clause: str, simplification: str, *, entities: list[str] | None = None) -> Analysis:
    entities = entities if entities is not None else extract_entities(clause)
    presence = check_entity_presence(entities, simplification)
    findings = run_detectors(clause, simplification, entities)
    warnings = [
        {"detector": name, "message": WARNING_TEXT[name], "detail": f.detail}
        for name, f in findings.items()
        if f.flagged
    ]
    if not simplification.strip():
        warnings.insert(
            0, {"detector": "empty", "message": "The model returned no text.", "detail": ""}
        )
    return Analysis(
        entities=[{"text": e, "kept": bool(presence.get(e, False))} for e in entities],
        warnings=warnings,
        reading_grade={"clause": _grade(clause), "simplification": _grade(simplification)},
        missing_entities=[e for e in entities if not presence.get(e, False)],
    )
