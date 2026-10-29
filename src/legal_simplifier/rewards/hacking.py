"""Reward-hacking detectors for clause simplifications.

Each detector looks for one way a policy can raise its reward without doing
the task:

* ``truncation``: very short output or very short sentences (cheap reading
  grade) that drop obligations, permissions or prohibitions the clause states.
* ``verbatim_copy``: the clause returned (nearly) unchanged, which keeps every
  entity and the meaning for free.
* ``word_salad``: the required entities strung together with little else, or
  repeated, to max out entity recall.
* ``modality_flip``: a "must" that became "may", a "must not" that lost its
  negation, or a prohibition that was not in the clause.

They are heuristics on surface text, so they are deterministic and cheap enough
to run inside the reward (the ``format`` reward subtracts a penalty per
detector that fires) and over saved outputs (``lcs detect-hacking``). They
will miss paraphrased failures; the NLI ``meaning`` reward and the LLM judge
(Phase 4) cover those.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from legal_simplifier.data.entities import check_entity_presence, normalise_for_match

DETECTORS = ("truncation", "verbatim_copy", "word_salad", "modality_flip")

_SUBJECT = r"(?:party|parties|one|person|\w+)"
PROHIBITION = re.compile(
    r"\b(?:must|shall|will|may|can|could|should)\s+not\b"
    r"|\b(?:cannot|can't|mustn't|shan't|won't|may\s+never|must\s+never|shall\s+never)\b"
    r"|\b(?:is|are)\s+(?:prohibited|forbidden|barred)\b"
    rf"|\b(?:no|neither)\s+{_SUBJECT}(?:\s+\w+)?\s+(?:shall|may|will|can|must)\b"
    r"|\bnot\s+(?:be\s+)?(?:permitted|allowed|entitled)\b",
    re.IGNORECASE,
)
OBLIGATION = re.compile(
    r"\b(?:must|shall|will|has\s+to|have\s+to|is\s+required\s+to|are\s+required\s+to"
    r"|agrees?\s+to|undertakes?\s+to|is\s+obligated\s+to|are\s+obligated\s+to"
    r"|is\s+responsible\s+for|are\s+responsible\s+for)\b",
    re.IGNORECASE,
)
PERMISSION = re.compile(
    r"\b(?:may|can|is\s+entitled\s+to|are\s+entitled\s+to|has\s+the\s+right\s+to"
    r"|have\s+the\s+right\s+to|is\s+permitted\s+to|are\s+permitted\s+to"
    r"|is\s+allowed\s+to|are\s+allowed\s+to|at\s+its\s+(?:sole\s+)?(?:option|discretion))\b",
    re.IGNORECASE,
)

_TOKEN = re.compile(r"[a-z0-9$%]+(?:[.'][a-z0-9]+)*")
_SENTENCE_END = re.compile(r"[.!?;:]+(?:\s+|$)")


@dataclass(frozen=True)
class Finding:
    detector: str
    flagged: bool
    detail: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(normalise_for_match(text))


def modality(text: str) -> Counter[str]:
    """Count obligation, permission and prohibition statements (prohibitions first)."""

    counts: Counter[str] = Counter()
    rest = text
    for match in PROHIBITION.finditer(text):
        counts["prohibition"] += 1
        rest = rest.replace(match.group(0), " ", 1)
    counts["obligation"] = len(OBLIGATION.findall(rest))
    counts["permission"] = len(PERMISSION.findall(rest))
    return +counts


def _sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_END.split(text) if s.strip()]


def _ngrams(words: Sequence[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def copy_overlap(clause: str, output: str, n: int = 4) -> float:
    """Share of the output's word n-grams that also occur in the clause (unigrams if short)."""

    out, src = tokens(output), tokens(clause)
    if not out:
        return 0.0
    if len(out) < n:
        return sum(w in set(src) for w in out) / len(out)
    grams = _ngrams(out, n)
    return len(grams & _ngrams(src, n)) / len(grams)


def detect_truncation(
    clause: str,
    output: str,
    *,
    max_length_ratio: float = 0.5,
    min_mean_sentence_words: float = 5.0,
) -> Finding:
    """Output that drops modal statements while being much shorter or unnaturally choppy."""

    before, after = modality(clause), modality(output)
    dropped = {k: before[k] - after.get(k, 0) for k in before if before[k] > after.get(k, 0)}
    if not dropped:
        return Finding("truncation", False)
    ratio = len(tokens(output)) / max(1, len(tokens(clause)))
    sentences = _sentences(output)
    mean_words = sum(len(tokens(s)) for s in sentences) / len(sentences) if sentences else 0.0
    short = ratio < max_length_ratio
    choppy = len(sentences) > 1 and mean_words < min_mean_sentence_words
    if not (short or choppy):
        return Finding("truncation", False)
    reason = []
    if short:
        reason.append(f"{ratio:.0%} of the clause's length")
    if choppy:
        reason.append(f"{mean_words:.1f} words per sentence")
    lost = ", ".join(f"{n} {k}" for k, n in sorted(dropped.items()))
    return Finding("truncation", True, f"drops {lost}; {' and '.join(reason)}")


def detect_verbatim_copy(clause: str, output: str, *, threshold: float = 0.85) -> Finding:
    if not output.strip():
        return Finding("verbatim_copy", False)
    if normalise_for_match(clause) == normalise_for_match(output):
        return Finding("verbatim_copy", True, "identical to the clause")
    overlap = copy_overlap(clause, output)
    flagged = overlap >= threshold
    return Finding("verbatim_copy", flagged, f"{overlap:.0%} of 4-grams copied" if flagged else "")


def detect_word_salad(
    output: str,
    entities: Sequence[str],
    *,
    min_other_words: int = 4,
    min_other_share: float = 0.25,
    max_repeats: int = 3,
) -> Finding:
    """Entities kept but strung together (little other text), or repeated over and over."""

    present = [e for e, kept in check_entity_presence(entities, output).items() if kept]
    if len(present) < 2:
        return Finding("word_salad", False)
    text = normalise_for_match(output)
    repeats = {}
    for entity in present:
        needle = normalise_for_match(entity)
        count = len(re.findall(rf"(?<![\w$]){re.escape(needle)}(?!\w)", text))
        if count >= max_repeats:
            repeats[entity] = count
        text = re.sub(rf"(?<![\w$]){re.escape(needle)}(?!\w)", " ", text)
    other = [w for w in _TOKEN.findall(text) if any(c.isalpha() for c in w)]
    total = len(tokens(output))
    share = len(other) / total if total else 0.0
    if repeats:
        listed = ", ".join(f"{e!r} x{n}" for e, n in repeats.items())
        return Finding("word_salad", True, f"repeats {listed}")
    if len(other) < min_other_words or share < min_other_share:
        return Finding(
            "word_salad",
            True,
            f"{len(present)} entities with {len(other)} other words ({share:.0%} of the text)",
        )
    return Finding("word_salad", False)


def detect_modality_flip(clause: str, output: str) -> Finding:
    """A change in what the clause requires, allows or forbids."""

    before, after = modality(clause), modality(output)
    if not after:
        return Finding("modality_flip", False)  # nothing modal left: truncation's job
    flips = []
    if before["prohibition"] and not after["prohibition"]:
        flips.append("a prohibition lost its negation")
    if after["prohibition"] and not before["prohibition"]:
        flips.append("adds a prohibition the clause does not have")
    # Counts, not presence, so a clause that mixes "must" and "may" is still checked.
    if after["obligation"] < before["obligation"] and after["permission"] > before["permission"]:
        flips.append("an obligation became a permission (must -> may)")
    if after["permission"] < before["permission"] and after["obligation"] > before["obligation"]:
        flips.append("a permission became an obligation (may -> must)")
    return Finding("modality_flip", bool(flips), "; ".join(flips))


def run_detectors(
    clause: str,
    output: str,
    entities: Sequence[str] = (),
    *,
    copy_threshold: float = 0.85,
) -> dict[str, Finding]:
    return {
        "truncation": detect_truncation(clause, output),
        "verbatim_copy": detect_verbatim_copy(clause, output, threshold=copy_threshold),
        "word_salad": detect_word_salad(output, entities),
        "modality_flip": detect_modality_flip(clause, output),
    }
