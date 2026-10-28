"""Cross-check our SARI against Hugging Face ``evaluate``'s SARI, and pin where they differ.

Our ``sentence_sari`` ports the Xu et al. (2016) reference code. The HF metric
(``evaluate-metric/sari``) differs in three ways, documented in
docs/decisions.md (ADR-003):

1. **Empty-set default.** When an operation has nothing to score at some
   n-gram order (nothing kept, nothing deleted, nothing added), HF defines
   0/0 = 1; we, like the reference code, score 0. In practice this bites when
   a prediction shares no 3- or 4-gram with the source (common for heavy
   rewrites), which makes HF's score higher than ours.
2. **Keep recall.** HF applies an "alleged bug fix": sum of good keep counts
   over the sum of reference-keep counts, instead of the mean per-n-gram
   ratio. The two agree when no n-gram repeats.
3. **Scale and tokenisation.** HF reports 0-100 after sacrebleu 13a
   tokenisation; we report 0-1 after a lowercase word/punctuation split.

On inputs where none of these applies, the two must agree exactly.
Needs the dev-only ``evaluate``/``sacrebleu``/``sacremoses`` packages and
downloads the pinned metric script once.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest

from legal_simplifier.rewards.sari import _ngrams, sentence_sari, tokenize

HF_SARI_REVISION = "1f05a19b3a4266ddaeb58fa506e1314d565f0c8e"

# (source, prediction, reference). Lowercase words and sentence punctuation only,
# so 13a and our tokenizer split identically, and no n-gram repeats.
AGREEING_EXAMPLES = [
    (
        "licensee shall remit payment within thirty days after receiving each invoice .",
        "licensee must pay within thirty days after receiving each invoice .",
        "licensee must pay within thirty days after it gets each invoice .",
    ),
    (
        "this agreement shall be governed by new york law without regard to conflicts principles .",
        "this agreement is governed by new york law .",
        "governed by new york law , this agreement ignores conflicts rules .",
    ),
    (
        "either party may terminate this agreement for convenience upon ninety days prior written notice .",
        "either party may terminate this deal for any reason upon ninety days prior written notice .",
        "either party may terminate it for any reason by giving ninety days written notice .",
    ),
    (
        "neither party may assign this agreement without prior written consent from its counterparty .",
        "neither party may assign this agreement unless its counterparty agrees in writing .",
        "neither party may assign this contract unless its counterparty first agrees in writing .",
    ),
    (
        "distributor shall not sell competing products during the term and twelve months thereafter .",
        "distributor must not sell competing products during the term or twelve months after .",
        "distributor must not sell competing products while this deal lasts or one year after .",
    ),
    (
        "all intellectual property created by consultant shall vest in company upon creation .",
        "company owns all intellectual property created by consultant , from creation .",
        "company owns all intellectual property created by consultant immediately .",
    ),
    (
        "supplier shall be exclusive provider of products within territory commencing january 2020 .",
        "from january 2020 , supplier is exclusive provider of products in territory .",
        "starting january 2020 , only supplier may be exclusive provider of products there .",
    ),
    (
        "upon expiration , vendor shall continue providing transition services for sixty days .",
        "after expiration , vendor must keep providing transition services for sixty days .",
        "when it ends , vendor must keep giving transition help for sixty days .",
    ),
    (
        "customer shall pay liquidated damages equal to five percent of fees for each late week .",
        "customer must pay liquidated damages equal to five percent of fees per late week .",
        "for every late week , customer pays liquidated damages equal to five percent .",
    ),
    (
        "any change of control at reseller shall be deemed an assignment requiring consent .",
        "a change of control at reseller counts as an assignment needing consent .",
        "if reseller has a change of control at board level , that needs consent .",
    ),
]


@pytest.fixture(scope="module")
def hf_sari() -> Any:
    evaluate = pytest.importorskip("evaluate")
    pytest.importorskip("sacrebleu")
    pytest.importorskip("sacremoses")
    return evaluate.load("sari", revision=HF_SARI_REVISION)


def _hf(metric: Any, source: str, prediction: str, reference: str) -> float:
    return (
        metric.compute(sources=[source], predictions=[prediction], references=[[reference]])["sari"]
        / 100
    )


def _preconditions_hold(source: str, prediction: str, reference: str) -> bool:
    """True when neither empty-set defaults nor repeated n-grams can make the two differ."""

    s, p, r = tokenize(source), tokenize(prediction), tokenize(reference)
    for n in range(1, 5):
        sg, pg, rg = (Counter(_ngrams(t, n)) for t in (s, p, r))
        if any(c > 1 for counter in (sg, pg, rg) for c in counter.values()):
            return False
        keep, keep_all = sg & pg, sg & rg
        delete = sg - pg
        added, added_all = set(pg) - set(sg), set(rg) - set(sg)
        if not (keep and keep_all and delete and added and added_all):
            return False
    return True


def test_agreeing_examples_satisfy_the_preconditions() -> None:
    assert len(AGREEING_EXAMPLES) == 10
    for example in AGREEING_EXAMPLES:
        assert _preconditions_hold(*example), example


@pytest.mark.slow
@pytest.mark.parametrize("example", AGREEING_EXAMPLES, ids=[f"ex{i}" for i in range(10)])
def test_matches_hf_sari_when_no_documented_difference_applies(
    hf_sari: Any, example: tuple[str, str, str]
) -> None:
    source, prediction, reference = example
    assert sentence_sari(source, prediction, [reference]) == pytest.approx(
        _hf(hf_sari, source, prediction, reference), abs=1e-9
    )


@pytest.mark.slow
def test_difference_empty_sets_identity_case(hf_sari: Any) -> None:
    # Prediction = source = reference: nothing deleted or added at any order.
    # HF: keep 1, delete 1 (0/0), add 1 (0/0) -> 1.0. Ours: 1, 0, 0 -> 1/3.
    text = "the tenant must pay rent on the first day of each month ."
    assert _hf(hf_sari, text, text, text) == pytest.approx(1.0)
    assert sentence_sari(text, text, [text]) == pytest.approx(1 / 3)


@pytest.mark.slow
def test_difference_empty_sets_pure_deletion(hf_sari: Any) -> None:
    # A deletion-only rewrite adds no new *words*, and neither does the reference, so
    # at n=1 the add term is 0/0: HF scores it 1, we score 0. (Deleting words still
    # creates new 2- to 4-grams, so only one of the four orders differs.) The gap
    # is therefore (1/4) / 3 = 1/12 of the scale.
    source = "the tenant must promptly pay the full monthly rent to landlord ."
    reference = "the tenant must pay the rent to landlord ."
    ours = sentence_sari(source, reference, [reference])
    theirs = _hf(hf_sari, source, reference, reference)
    assert theirs - ours == pytest.approx(1 / 12)


@pytest.mark.slow
def test_difference_keep_recall_with_repeated_ngrams(hf_sari: Any) -> None:
    # Repeated n-grams make the two keep-recall formulas diverge even though every
    # operation has something to score.
    source = "the party shall pay the fee and the party shall file the report ."
    prediction = "the party must pay the fee and file the report ."
    reference = "the party pays the fee and the party files the report ."
    assert not _preconditions_hold(source, prediction, reference)
    ours = sentence_sari(source, prediction, [reference])
    theirs = _hf(hf_sari, source, prediction, reference)
    assert ours != pytest.approx(theirs, abs=1e-6)
