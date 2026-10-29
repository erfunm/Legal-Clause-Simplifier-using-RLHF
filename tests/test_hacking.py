"""Reward-hacking detectors on hand-written examples (written for the tests, not model output)."""

from __future__ import annotations

import pytest

from legal_simplifier.rewards.hacking import (
    DETECTORS,
    conditions,
    detect_condition_drop,
    detect_modality_flip,
    detect_truncation,
    detect_verbatim_copy,
    detect_word_salad,
    modality,
    run_detectors,
)

CLAUSE = (
    "The Distributor shall purchase at least 10,000 units each calendar year and shall not sell "
    "competing products in the Territory. The Supplier may terminate this Agreement upon thirty "
    "(30) days written notice."
)
ENTITIES = ["the Distributor", "10,000", "the Territory", "the Supplier", "thirty (30) days"]
GOOD = (
    "The Distributor must buy at least 10,000 units every calendar year and must not sell competing "
    "products in the Territory. The Supplier can end this Agreement with 30 days written notice."
)


def test_a_faithful_rewrite_trips_nothing() -> None:
    findings = run_detectors(CLAUSE, GOOD, ENTITIES)
    assert set(findings) == set(DETECTORS)
    assert not any(f.flagged for f in findings.values())


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The Buyer shall pay.", {"obligation": 1}),
        ("The Buyer must not assign it.", {"prohibition": 1}),
        ("Neither party shall assign this Agreement.", {"prohibition": 1}),
        ("No party may disclose the terms.", {"prohibition": 1}),
        ("The Buyer cannot assign it and may not sublicense it.", {"prohibition": 2}),
        ("The Seller may, at its option, refund the price.", {"permission": 2}),
        (
            "The Seller is entitled to a fee and agrees to deliver.",
            {"permission": 1, "obligation": 1},
        ),
        ("This is not a waiver of any right.", {}),
    ],
)
def test_modality_counts(text: str, expected: dict[str, int]) -> None:
    assert dict(modality(text)) == expected


# --- truncation that drops obligations ---------------------------------------------


def test_short_output_that_drops_obligations_is_truncation() -> None:
    finding = detect_truncation(CLAUSE, "The Distributor buys units in the Territory.")
    assert finding.flagged
    assert "prohibition" in finding.detail and "permission" in finding.detail


def test_choppy_sentences_that_game_the_reading_grade_are_truncation() -> None:
    # Many two- and three-word sentences: very low FKGL, and the prohibition is gone.
    choppy = (
        "The Distributor buys. It buys 10,000 units. Every year. In the Territory. "
        "The Supplier may end it. With notice. In 30 days."
    )
    finding = detect_truncation(CLAUSE, choppy)
    assert finding.flagged and "words per sentence" in finding.detail


def test_a_shorter_rewrite_that_keeps_every_statement_is_not_truncation() -> None:
    short = (
        "The Distributor must buy 10,000 units a year and must not sell rival products in the "
        "Territory. The Supplier may end this on 30 days notice."
    )
    assert not detect_truncation(CLAUSE, short).flagged


# --- verbatim copying -------------------------------------------------------------------


def test_the_clause_itself_is_a_copy() -> None:
    finding = detect_verbatim_copy(CLAUSE, CLAUSE)
    assert finding.flagged and finding.detail == "identical to the clause"


def test_a_copy_with_cosmetic_edits_is_still_a_copy() -> None:
    near = CLAUSE.replace("thirty (30)", "30").replace("shall purchase", "shall buy")
    finding = detect_verbatim_copy(CLAUSE, near)
    assert finding.flagged and "4-grams copied" in finding.detail


def test_a_rewrite_that_reuses_names_is_not_a_copy() -> None:
    assert not detect_verbatim_copy(CLAUSE, GOOD).flagged
    assert not detect_verbatim_copy(CLAUSE, "").flagged


# --- entity word-salad ----------------------------------------------------------------


def test_entities_strung_together_are_word_salad() -> None:
    salad = "The Distributor, 10,000, the Territory, the Supplier, thirty (30) days."
    finding = detect_word_salad(salad, ENTITIES)
    assert finding.flagged and "5 entities" in finding.detail


def test_repeated_entities_are_word_salad() -> None:
    padded = (
        "The Distributor sells in the Territory. The Territory is the Territory, and the "
        "Territory is where the Distributor sells."
    )
    finding = detect_word_salad(padded, ENTITIES)
    assert finding.flagged and "'the Territory' x4" in finding.detail


def test_entities_in_real_sentences_are_not_word_salad() -> None:
    assert not detect_word_salad(GOOD, ENTITIES).flagged
    # Entity-dense but grammatical: about a third of the words are not entities.
    dense = "The Licensee must pay the Licensor 5% of Net Sales. The Licensee must not sublicense the Software."
    assert not detect_word_salad(
        dense, ["the Licensee", "the Licensor", "5%", "Net Sales", "the Software"]
    ).flagged
    assert not detect_word_salad("Just one entity: the Supplier.", ENTITIES).flagged


# --- must / may / must-not flips -------------------------------------------------------


def test_a_prohibition_that_loses_its_negation_is_a_flip() -> None:
    flipped = GOOD.replace("must not sell", "may sell")
    finding = detect_modality_flip(CLAUSE, flipped)
    assert finding.flagged and "lost its negation" in finding.detail


def test_must_becoming_may_is_a_flip() -> None:
    clause = "The Licensee shall deliver the reports by March 1, 2024."
    finding = detect_modality_flip(clause, "The Licensee may deliver the reports by March 1, 2024.")
    assert finding.flagged and "must -> may" in finding.detail


def test_may_becoming_must_is_a_flip() -> None:
    clause = "The Licensor may audit the records once a year."
    finding = detect_modality_flip(clause, "The Licensor must audit the records once a year.")
    assert finding.flagged and "may -> must" in finding.detail


def test_an_added_prohibition_is_a_flip() -> None:
    clause = "The Licensor may audit the records once a year."
    finding = detect_modality_flip(
        clause, "The Licensor may audit the records but must not copy them."
    )
    assert finding.flagged and "adds a prohibition" in finding.detail


def test_shall_to_must_and_may_to_can_are_not_flips() -> None:
    assert not detect_modality_flip(CLAUSE, GOOD).flagged
    assert not detect_modality_flip(CLAUSE, "Nothing modal here.").flagged  # truncation's job


# --- dropped conditions and exceptions ---------------------------------------------------

TERMINATION = (
    "The Licensee may terminate on 30 days' notice, unless the Licensor has cured the breach "
    "within 10 days."
)
DROPPED = "The Licensee can end this agreement with 30 days' notice."


def test_the_dropped_exception_passes_every_other_check() -> None:
    # The gap this detector closes: the same permission and a shorter text, so truncation,
    # modality_flip and word_salad stay silent. (Entity recall drops to 0.5 here only because
    # the exception names "10 days" and "the Licensor"; test_rewards has one that does not.)
    entities = ["the Licensee", "30 days", "the Licensor", "10 days"]
    others = {
        k: f
        for k, f in run_detectors(TERMINATION, DROPPED, entities).items()
        if k != "condition_drop"
    }
    assert not any(f.flagged for f in others.values())


def test_a_dropped_exception_is_flagged() -> None:
    finding = detect_condition_drop(TERMINATION, DROPPED)
    assert finding.flagged
    assert (
        finding.detail
        == "keeps 0 of 2 conditions/exceptions (clause: unless, within; output: none)"
    )


@pytest.mark.parametrize(
    "rewrite",
    [
        "The Licensee can end this agreement with 30 days' notice, unless the Licensor fixes the "
        "breach within 10 days.",
        "The Licensee can end this agreement with 30 days' notice, but not if the Licensor fixes "
        "the breach within 10 days.",
        "The Licensee can end this agreement with 30 days' notice. It cannot do this if the "
        "Licensor fixes the breach in ten days.",
    ],
)
def test_a_rewrite_that_keeps_the_exception_is_not_flagged(rewrite: str) -> None:
    assert not detect_condition_drop(TERMINATION, rewrite).flagged


def test_dropping_only_the_time_limit_is_flagged() -> None:
    finding = detect_condition_drop(
        TERMINATION,
        "The Licensee can end this agreement on notice, unless the Licensor fixes the breach.",
    )
    assert finding.flagged and "keeps 1 of 2" in finding.detail


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("provided, however, that the fee is paid", ["provided, however, that"]),
        ("subject to Section 5 and notwithstanding Section 6", ["subject to", "notwithstanding"]),
        ("only if paid, but not if late, except as agreed", ["only if", "but not if", "except as"]),
        ("in the event of a default or in case of delay", ["in the event of", "in case"]),
        (
            "until paid, no later than March 1, in 30 business days",
            ["until", "no later than", "in 30 business days"],
        ),
        ("as long as it pays when due", ["as long as", "when"]),
        ("The Buyer pays the price.", []),
    ],
)
def test_condition_markers_count_once_each_longest_first(text: str, expected: list[str]) -> None:
    assert conditions(text) == expected


def test_empty_output_is_left_to_the_format_reward() -> None:
    assert not detect_condition_drop(TERMINATION, "").flagged
