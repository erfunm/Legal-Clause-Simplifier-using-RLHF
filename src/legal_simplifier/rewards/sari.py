"""Sentence-level SARI (Xu et al., 2016, "Optimizing Statistical Machine Translation for Text Simplification").

``easse`` is not published on PyPI, so this is a direct port of the reference
``SARIngram`` / ``SARIsent`` algorithm: for n = 1..4 it scores n-grams the
system kept (F1), deleted (precision) and added (F1) against the source and
references, then averages the three. Tokenisation is lowercase word/punctuation
splitting, so absolute values are consistent within this project but not
directly comparable with EASSE's corpus-level SARI.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

_TOKEN = re.compile(r"\w+|[^\w\s]")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _ngrams(tokens: Sequence[str], n: int) -> list[str]:
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _sari_ngram(
    source: list[str], prediction: list[str], references: list[list[str]]
) -> tuple[float, float, float]:
    num_refs = len(references)
    ref_counter = Counter(gram for ref in references for gram in ref)
    src_counter = Counter(source)
    pred_counter = Counter(prediction)
    src_rep = Counter({g: c * num_refs for g, c in src_counter.items()})
    pred_rep = Counter({g: c * num_refs for g, c in pred_counter.items()})

    # Keep: n-grams in both source and prediction, rewarded when references keep them too.
    keep = src_rep & pred_rep
    keep_good = keep & ref_counter
    keep_all = src_rep & ref_counter
    keep_p = sum(keep_good[g] / keep[g] for g in keep) / len(keep) if keep else 0.0
    # Kept n-grams absent from every reference give 0/0 in the reference code; count them as 0.
    keep_r = (
        sum(keep_good[g] / keep_all[g] for g in keep if keep_all[g]) / len(keep_all)
        if keep_all
        else 0.0
    )
    keep_score = _f1(keep_p, keep_r)

    # Delete: n-grams of the source the prediction dropped; only precision is used.
    delete = src_rep - pred_rep
    delete_good = delete - ref_counter
    delete_p = sum(delete_good[g] / delete[g] for g in delete) / len(delete) if delete else 0.0

    # Add: new n-grams in the prediction, rewarded when references add them too.
    added = set(pred_counter) - set(src_counter)
    added_good = added & set(ref_counter)
    added_all = set(ref_counter) - set(src_counter)
    add_p = len(added_good) / len(added) if added else 0.0
    add_r = len(added_good) / len(added_all) if added_all else 0.0
    add_score = _f1(add_p, add_r)

    return keep_score, delete_p, add_score


def sentence_sari(source: str, prediction: str, references: Sequence[str]) -> float:
    """Return SARI in [0, 1] for one prediction against one or more references."""

    if not references:
        raise ValueError("SARI needs at least one reference")
    src_tokens = tokenize(source)
    pred_tokens = tokenize(prediction)
    ref_tokens = [tokenize(ref) for ref in references]

    keep_scores, delete_scores, add_scores = [], [], []
    for n in range(1, 5):
        keep, delete, add = _sari_ngram(
            _ngrams(src_tokens, n), _ngrams(pred_tokens, n), [_ngrams(r, n) for r in ref_tokens]
        )
        keep_scores.append(keep)
        delete_scores.append(delete)
        add_scores.append(add)
    return (sum(keep_scores) + sum(delete_scores) + sum(add_scores)) / 12
