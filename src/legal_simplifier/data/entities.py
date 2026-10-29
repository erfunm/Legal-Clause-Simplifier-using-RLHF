"""Extract the entities a simplification must keep, and check that it kept them.

Extraction is regex-based by default so it is deterministic and testable; spaCy
NER can be layered on top (``use_spacy=True``, needs the ``nlp`` extra and an
installed English model). Matching is done on a normalised form so harmless
reformatting ("USD 1,000" vs "$1000", "thirty (30) days" vs "30 days",
"$1,000,000" vs "$1 million" vs "one million dollars") still counts as preserved.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from functools import lru_cache
from typing import Any

LOGGER = logging.getLogger(__name__)

_MONTH = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
    r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?"
)
_ORDINAL = r"(?:st|nd|rd|th)?"
_CURRENCY = r"(?:US\$|USD\s?|\$|€|£|EUR\s?|GBP\s?)"
_NUMBER = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_SCALE = r"(?:\s(?:thousand|million|billion))?"
_COMPANY_SUFFIX = (
    r"(?:Inc\.?|Incorporated|LLC|L\.L\.C\.|Ltd\.?|Limited|Corporation|Corp\.?|Co\.|"
    r"L\.P\.|LP|LLP|PLC|plc|GmbH|AG|S\.A\.|N\.V\.|B\.V\.)"
)

DATE_PATTERNS = [
    re.compile(rf"\b{_MONTH}\s+\d{{1,2}}{_ORDINAL},?\s+\d{{4}}\b"),
    re.compile(rf"\b\d{{1,2}}{_ORDINAL}\s+(?:day\s+of\s+)?{_MONTH},?\s+\d{{4}}\b"),
    re.compile(rf"\b{_MONTH}\s+\d{{4}}\b"),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
]
MONEY_PATTERNS = [
    re.compile(rf"{_CURRENCY}\s?(?:{_NUMBER}){_SCALE}"),
    re.compile(rf"\b(?:{_NUMBER}){_SCALE}\s(?:dollars|USD|euros|pounds)\b", re.IGNORECASE),
]
PERCENT_PATTERNS = [
    re.compile(r"\b\d+(?:\.\d+)?\s?(?:%|percent\b)", re.IGNORECASE),
]
DURATION_PATTERNS = [
    re.compile(
        r"\b(?:[A-Za-z]+(?:-[A-Za-z]+)?\s+\(\d+\)|\d+)\s+"
        r"(?:business\s+|calendar\s+|consecutive\s+)?(?:days?|weeks?|months?|years?)\b",
        re.IGNORECASE,
    ),
]
# Quoted, capitalised defined terms: (the "Licensee"), (“Effective Date”).
DEFINED_TERM_PATTERN = re.compile(r"[\"“]([A-Z][A-Za-z0-9&'\-]*(?:\s+[A-Za-z0-9&'\-]+){0,5})[\"”]")
# Capitalised names ending in a company suffix: Acme Widgets, Inc.
PARTY_PATTERN = re.compile(
    rf"\b(?:[A-Z][A-Za-z0-9&'.\-]*\s+){{0,5}}[A-Z][A-Za-z0-9&'.\-]*,?\s+{_COMPANY_SUFFIX}"
)
# Runs of capitalised words; mid-sentence these are defined terms or names
# (the Territory, the Effective Date, New York).
CAPITALISED_PATTERN = re.compile(r"\b[A-Z][A-Za-z'\-]*(?:\s+[A-Z][A-Za-z'\-]*)*")

# Capitalised only because they start a sentence or phrase.
LEADING_STOPWORDS = frozenset(
    [
        "A",
        "An",
        "All",
        "Any",
        "At",
        "Before",
        "By",
        "During",
        "Each",
        "Either",
        "For",
        "From",
        "His",
        "If",
        "In",
        "Its",
        "Neither",
        "No",
        "Notwithstanding",
        "Nothing",
        "On",
        "Subject",
        "Such",
        "That",
        "The",
        "Their",
        "These",
        "This",
        "Those",
        "To",
        "Unless",
        "Until",
        "Upon",
        "When",
        "Where",
        "Whereas",
        "Within",
    ]
)
# Capitalised everywhere in contracts but not worth enforcing.
GENERIC_TERMS = frozenset(
    [
        "Agreement",
        "Appendix",
        "Article",
        "Clause",
        "Exhibit",
        "Parties",
        "Party",
        "Schedule",
        "Section",
        "State",
    ]
)

SPACY_LABELS = frozenset({"DATE", "MONEY", "ORG", "PERSON", "GPE", "PERCENT", "LAW"})


def _strip_leading_stopwords(words: list[str]) -> list[str]:
    while words and words[0] in LEADING_STOPWORDS:
        words = words[1:]
    return words


def _starts_sentence(text: str, start: int) -> bool:
    before = text[:start].rstrip()
    return not before or before[-1] in ".!?:;"


def _capitalised_terms(text: str) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for match in CAPITALISED_PATTERN.finditer(text):
        words = match.group().split()
        if all(w.isupper() and len(w) > 1 for w in words):
            continue  # SHOUTED boilerplate such as "IN NO EVENT SHALL"
        if _starts_sentence(text, match.start()):
            words = words[1:]
        words = _strip_leading_stopwords(words)
        term = " ".join(words)
        if term and term not in GENERIC_TERMS:
            found.append((match.start(), term))
    return found


def _regex_entities(text: str) -> list[str]:
    found: list[tuple[int, str]] = []
    for pattern in (*DATE_PATTERNS, *MONEY_PATTERNS, *PERCENT_PATTERNS, *DURATION_PATTERNS):
        found.extend((m.start(), m.group().strip()) for m in pattern.finditer(text))
    found.extend((m.start(1), m.group(1).strip()) for m in DEFINED_TERM_PATTERN.finditer(text))
    for match in PARTY_PATTERN.finditer(text):
        name = " ".join(_strip_leading_stopwords(match.group().strip().split()))
        if name:
            found.append((match.start(), name))
    found.extend(_capitalised_terms(text))
    return [value for _, value in sorted(found, key=lambda item: item[0])]


def _drop_subsumed(entities: list[str]) -> list[str]:
    """Drop entities contained in a longer one ("January" inside "January 1, 2020")."""

    keys = [normalise_for_match(e) for e in entities]
    kept: list[str] = []
    for entity, key in zip(entities, keys, strict=True):
        pattern = re.compile(rf"(?<![\w$]){re.escape(key)}(?!\w)")
        if not any(other != key and pattern.search(other) for other in keys):
            kept.append(entity)
    return kept


@lru_cache(maxsize=1)
def _load_spacy(model: str) -> Any | None:
    try:
        import spacy
    except ImportError:
        LOGGER.warning("spaCy is not installed; using regex-only entity extraction.")
        return None
    try:
        return spacy.load(model)
    except OSError:
        LOGGER.warning("spaCy model %r is not installed; using regex-only extraction.", model)
        return None


def extract_entities(
    text: str, *, use_spacy: bool = False, spacy_model: str = "en_core_web_sm"
) -> list[str]:
    """Return entities in ``text`` in order of appearance, without duplicates.

    Covers dates, money, percentages, durations, defined terms (quoted, or
    capitalised mid-sentence) and company names. With ``use_spacy`` the spaCy
    NER entities are added first.
    """

    entities: list[str] = []
    if use_spacy:
        nlp = _load_spacy(spacy_model)
        if nlp is not None:
            entities.extend(
                ent.text.strip() for ent in nlp(text).ents if ent.label_ in SPACY_LABELS
            )
    entities.extend(_regex_entities(text))
    return _drop_subsumed(dedupe_entities(entities))


def dedupe_entities(entities: Iterable[str]) -> list[str]:
    """Strip entities and drop empty or duplicate ones (by normalised form), keeping order."""

    cleaned: list[str] = []
    seen: set[str] = set()
    for entity in entities:
        entity = (entity or "").strip()
        key = normalise_for_match(entity)
        if key and key not in seen:
            seen.add(key)
            cleaned.append(entity)
    return cleaned


_NUMBER_WORD_WITH_DIGITS = re.compile(r"\b[a-z]+(?:-[a-z]+)?\s*\((\d+)\)")

_UNITS = {
    w: i
    for i, w in enumerate(
        [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
        ]
    )
}
_TENS = {
    w: 10 * i
    for i, w in enumerate(
        ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"], 2
    )
}
_SCALES = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}
_WORD_OR_GAP = re.compile(r"[a-z]+|[\s-]+|[^a-z\s-]+")


def _number_words_value(words: list[str]) -> int:
    total, current = 0, 0
    for word in words:
        if word in _UNITS:
            current += _UNITS[word]
        elif word in _TENS:
            current += _TENS[word]
        elif word == "hundred":
            current = max(current, 1) * 100
        else:
            total += max(current, 1) * _SCALES[word]
            current = 0
    return total + current


def _replace_number_words(text: str) -> str:
    """ "twenty-five" -> "25", "one hundred and twenty" -> "120", "two million" -> "2000000".

    A run must start with a unit or tens word, so a scale word after digits
    ("1.5 million") is left for the scale step.
    """

    out: list[str] = []
    run: list[str] = []
    pending_gap = ""

    def flush() -> None:
        nonlocal pending_gap
        if run:
            out.append(str(_number_words_value(run)))
            run.clear()
        out.append(pending_gap)
        pending_gap = ""

    for token in _WORD_OR_GAP.findall(text):
        is_word = token[0].isalpha()
        if is_word and (token in _UNITS or token in _TENS) and (not run or run[-1] not in _UNITS):
            if not run:
                flush()
            run.append(token)
        elif is_word and run and (token == "hundred" or token in _SCALES):
            run.append(token)
        elif is_word and run and token == "and" and run[-1] == "hundred":
            continue
        elif not is_word and run and token.strip(" -") == "":
            pending_gap = token
            continue
        else:
            flush()
            out.append(token)
            continue
        pending_gap = ""
    flush()
    return "".join(out)


def _apply_scale(match: re.Match[str]) -> str:
    value = float(match.group(2)) * _SCALES[match.group(3)]
    return f"{match.group(1)}{int(value) if value.is_integer() else value}"


def normalise_for_match(text: str) -> str:
    """Canonicalise formatting so equivalent spellings of an entity compare equal."""

    text = text.lower()
    text = text.translate(
        str.maketrans({"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"})
    )
    text = _NUMBER_WORD_WITH_DIGITS.sub(r"\1", text)  # "thirty (30)" -> "30"
    text = re.sub(r"\b(?:us\$|usd)\s?", "$", text)
    text = re.sub(r"\$\s+", "$", text)
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    text = _replace_number_words(text)
    text = re.sub(r"(\$?)(\d+(?:\.\d+)?)\s+(thousand|million|billion)\b", _apply_scale, text)
    text = re.sub(r"(?<![$\d.])(\d+(?:\.\d+)?)\s+(?:us\s+)?dollars?\b", r"$\1", text)
    text = re.sub(r"(\d)(?:st|nd|rd|th)\b", r"\1", text)
    text = re.sub(r"\s?percent\b", "%", text)
    text = re.sub(r"\s+%", "%", text)
    text = text.replace(",", " ")
    return " ".join(text.split())


def check_entity_presence(entities: Sequence[str], text: str) -> dict[str, bool]:
    """Map each entity to whether it appears in ``text`` (normalised, whole-token match)."""

    haystack = normalise_for_match(text)
    presence: dict[str, bool] = {}
    for entity in entities:
        needle = normalise_for_match(entity)
        if not needle:
            continue
        pattern = rf"(?<![\w$]){re.escape(needle)}(?!\w)"
        presence[entity] = re.search(pattern, haystack) is not None
    return presence


def entity_recall(entities: Sequence[str], text: str) -> float:
    """Fraction of ``entities`` preserved in ``text``; 1.0 when there is nothing to preserve."""

    presence = check_entity_presence(entities, text)
    if not presence:
        return 1.0
    return sum(presence.values()) / len(presence)
