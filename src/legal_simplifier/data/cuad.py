"""Extract clause spans from CUAD v1 (The Atticus Project, CC BY 4.0).

CUAD is SQuAD-formatted: each contract is one ``context`` (the *whole*
contract) with one question per category, and the clause text lives in that
question's ``answers`` as character spans. The old pipeline went wrong three
ways, all fixed here:

* ``load_dataset("cuad")`` relied on a dataset loading script, which current
  ``datasets`` no longer runs. We download ``CUAD_v1.json`` from the official
  Hub repo at a pinned revision and parse it directly.
* It used ``context`` (the full contract) as the clause. We use the answer spans.
* It matched categories against ``question``, which is a long prompt
  ("Highlight the parts (if any) of this contract related to ..."), so nothing
  ever matched. The category is the suffix of the question id:
  ``<contract title>__<Category>``.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from legal_simplifier.data.entities import extract_entities
from legal_simplifier.data.schema import ClauseCandidate
from legal_simplifier.prompts import normalise_clause

CUAD_REPO = "theatticusproject/cuad"
CUAD_FILE = "CUAD_v1/CUAD_v1.json"
CUAD_REVISION = "a3c393f5d103fd0c516374e4fdff676c8176dcb1"

# High-value clause types with plain-language obligations. Order sets the
# primary category when CUAD gives one span several labels.
DEFAULT_CATEGORIES: tuple[str, ...] = (
    "Governing Law",
    "Termination For Convenience",
    "Renewal Term",
    "Anti-Assignment",
    "Change Of Control",
    "Exclusivity",
    "Non-Compete",
    "Ip Ownership Assignment",
    "Cap On Liability",
    "Liquidated Damages",
    "Post-Termination Services",
)

# Text between two spans that still counts as one clause (whitespace, light punctuation).
_JOINABLE_GAP = re.compile(r"[\s,;:.()\-]{0,3}")


@dataclass
class ExtractionReport:
    """What extraction kept and filtered, for the stats file and dataset card."""

    categories: list[str]
    min_chars: int
    min_words: int
    max_chars: int
    n_contracts: int = 0
    n_answer_spans: int = 0
    n_after_merging_adjacent: int = 0
    n_unique_spans: int = 0
    """Distinct span texts per contract; equals n_too_short + n_too_long + n_kept."""
    n_multi_label: int = 0
    n_too_short: int = 0
    n_too_long: int = 0
    n_kept: int = 0
    kept_per_category: dict[str, int] = field(default_factory=dict)
    filtered_examples: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "categories": self.categories,
            "thresholds": {
                "min_chars": self.min_chars,
                "min_words": self.min_words,
                "max_chars": self.max_chars,
            },
            "n_contracts": self.n_contracts,
            "n_answer_spans": self.n_answer_spans,
            "n_after_merging_adjacent": self.n_after_merging_adjacent,
            "n_unique_spans": self.n_unique_spans,
            "n_multi_label": self.n_multi_label,
            "n_too_short": self.n_too_short,
            "n_too_long": self.n_too_long,
            "n_kept": self.n_kept,
            "kept_per_category": self.kept_per_category,
        }


def download_cuad(revision: str = CUAD_REVISION, cache_dir: Path | str | None = None) -> Path:
    """Download (or reuse the cached) ``CUAD_v1.json`` and return its path."""

    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            CUAD_REPO,
            CUAD_FILE,
            repo_type="dataset",
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    )


def load_cuad_json(path: Path | str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        data = json.load(handle)
    if "data" not in data:
        raise ValueError(f"{path} is not SQuAD-formatted CUAD JSON (no 'data' key)")
    return data


def category_from_qa_id(qa_id: str) -> str:
    """``"ACME_2019-EX-10__Governing Law"`` -> ``"Governing Law"``."""

    if "__" not in qa_id:
        raise ValueError(f"unexpected CUAD question id {qa_id!r}")
    return qa_id.rsplit("__", 1)[1]


def merge_spans(context: str, spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping spans and spans separated only by whitespace or light punctuation."""

    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if merged and (
            start <= merged[-1][1] or _JOINABLE_GAP.fullmatch(context[merged[-1][1] : start])
        ):
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def _spans_for(qa: dict[str, Any], context: str) -> list[tuple[int, int]]:
    spans = []
    for answer in qa.get("answers", []):
        start = int(answer["answer_start"])
        text = answer["text"]
        if context[start : start + len(text)] != text:
            raise ValueError(f"answer offset does not match context in {qa['id']}")
        spans.append((start, start + len(text)))
    return spans


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()


def extract_clauses(
    cuad: dict[str, Any],
    *,
    categories: Sequence[str] = DEFAULT_CATEGORIES,
    min_chars: int = 40,
    min_words: int = 6,
    max_chars: int = 1500,
    use_spacy: bool = False,
) -> tuple[list[ClauseCandidate], ExtractionReport]:
    """Turn CUAD answer spans in ``categories`` into clause candidates.

    Spans are merged per contract and category, identical spans that CUAD
    labelled with several categories become one candidate (primary category
    by ``categories`` order), and spans outside the length bounds are dropped
    and counted.
    """

    wanted = {c.lower(): c for c in categories}
    priority = {c: i for i, c in enumerate(categories)}
    report = ExtractionReport(list(categories), min_chars, min_words, max_chars)
    candidates: list[ClauseCandidate] = []
    per_category: Counter[str] = Counter()
    examples: dict[str, list[str]] = {"too_short": [], "too_long": []}

    for document in cuad["data"]:
        contract_id = document["title"]
        report.n_contracts += 1
        # text -> categories, in first-seen order
        by_text: dict[str, list[str]] = {}
        for paragraph in document["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                category = wanted.get(category_from_qa_id(qa["id"]).lower())
                if category is None:
                    continue
                spans = _spans_for(qa, context)
                report.n_answer_spans += len(spans)
                for start, end in merge_spans(context, spans):
                    report.n_after_merging_adjacent += 1
                    text = normalise_clause(context[start:end])
                    by_text.setdefault(text, [])
                    if category not in by_text[text]:
                        by_text[text].append(category)

        report.n_unique_spans += len(by_text)
        for index, (text, labels) in enumerate(by_text.items()):
            report.n_multi_label += len(labels) > 1
            if len(text) < min_chars or len(text.split()) < min_words:
                report.n_too_short += 1
                examples["too_short"].append(text)
                continue
            if len(text) > max_chars:
                report.n_too_long += 1
                examples["too_long"].append(text[:200] + "...")
                continue
            labels = sorted(labels, key=priority.__getitem__)
            candidates.append(
                ClauseCandidate(
                    id=f"{_slug(contract_id)[:80]}__{_slug(labels[0])}__{index:03d}",
                    contract_id=contract_id,
                    category=labels[0],
                    extra_categories=labels[1:],
                    clause=text,
                    entities=extract_entities(text, use_spacy=use_spacy),
                )
            )
            per_category[labels[0]] += 1

    report.n_kept = len(candidates)
    report.kept_per_category = {c: per_category.get(c, 0) for c in categories}
    report.filtered_examples = {k: v[:5] for k, v in examples.items()}
    return candidates, report
