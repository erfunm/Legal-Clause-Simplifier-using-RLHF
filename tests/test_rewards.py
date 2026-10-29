from __future__ import annotations

from collections.abc import Sequence

import pytest

from legal_simplifier.config import FormatConfig, RewardConfig, RewardWeights, SimplicityConfig
from legal_simplifier.rewards import components
from legal_simplifier.rewards.components import (
    bertscore,
    entity_preservation_score,
    fkgl,
    fkgl_band_score,
    format_score,
    sari,
    simplicity_score,
)
from legal_simplifier.rewards.sari import sentence_sari, tokenize
from legal_simplifier.rewards.suite import REWARD_NAMES, RewardSuite, completion_text


class FakeScorer:
    """Stands in for BERTScore: token-overlap F1, and records every call."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def score(self, predictions: Sequence[str], references: Sequence[str]) -> list[float]:
        self.calls.append(len(predictions))
        out = []
        for p, r in zip(predictions, references, strict=True):
            ps, rs = set(tokenize(p)), set(tokenize(r))
            out.append(2 * len(ps & rs) / (len(ps) + len(rs)) if ps or rs else 0.0)
        return out


# --- SARI -----------------------------------------------------------------


def test_sari_copying_the_source_when_reference_equals_source_is_one_third() -> None:
    # Known property: all keep n-grams are correct, nothing deleted or added.
    text = "the tenant must pay rent on the first day of each month"
    assert sentence_sari(text, text, [text]) == pytest.approx(1 / 3)


def test_sari_is_one_when_prediction_matches_the_reference() -> None:
    # Needs kept, deleted and added n-grams at every order n = 1..4.
    source = "the tenant must pay the rent forthwith upon receipt of the invoice"
    reference = "the tenant must pay the rent as soon as the invoice arrives"
    assert sentence_sari(source, reference, [reference]) == pytest.approx(1.0)


def test_sari_keep_term_is_zero_at_orders_with_nothing_kept() -> None:
    # Only the unigram "the" survives, so keep scores 1 at n=1 and 0 at n=2..4:
    # SARI = (1/4 + 1 + 1) / 3.
    source = "the lessee shall remit payment forthwith"
    reference = "the renter must pay now"
    assert sentence_sari(source, reference, [reference]) == pytest.approx(0.75)


def test_sari_prefers_the_reference_over_copying_or_deleting_everything() -> None:
    source = "the lessee shall remit payment forthwith upon receipt of an invoice"
    reference = "the renter must pay as soon as they get an invoice"
    good = sentence_sari(source, reference, [reference])
    copy = sentence_sari(source, source, [reference])
    empty = sentence_sari(source, "", [reference])
    assert good > copy
    assert good > empty


def test_sari_is_bounded_and_needs_references() -> None:
    score = sentence_sari("a b c d", "a x c y", ["a b x"])
    assert 0.0 <= score <= 1.0
    with pytest.raises(ValueError):
        sentence_sari("a", "a", [])


def test_batched_sari_matches_sentence_sari() -> None:
    sources, preds, refs = ["a b c", "d e f"], ["a c", "d x"], ["a c", "d e"]
    assert sari(sources, preds, refs) == [
        sentence_sari(s, p, [r]) for s, p, r in zip(sources, preds, refs, strict=True)
    ]


# --- BERTScore wrapper ---------------------------------------------------------


def test_bertscore_scores_the_whole_batch_in_one_call() -> None:
    scorer = FakeScorer()
    scores = bertscore(["a b", "c d", "e f"], ["a b", "c x", "y z"], scorer)
    assert scorer.calls == [3]
    assert scores[0] == pytest.approx(1.0)
    assert scores[2] == pytest.approx(0.0)


def test_bertscore_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        bertscore(["a"], ["a", "b"], FakeScorer())


def test_bertscore_model_is_loaded_once(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[str] = []

    class StubScorer:
        def __init__(self, model_type: str, **_: object) -> None:
            created.append(model_type)

    components.get_bertscore_scorer.cache_clear()
    monkeypatch.setattr(components, "BertScoreScorer", StubScorer)
    try:
        first = components.get_bertscore_scorer("roberta-large")
        second = components.get_bertscore_scorer("roberta-large")
    finally:
        components.get_bertscore_scorer.cache_clear()
    assert first is second
    assert created == ["roberta-large"]


# --- simplicity -------------------------------------------------------------

PLAIN = (  # FKGL about 6.8: inside the default grade band
    "The Licensee must pay the Licensor 5% of Net Sales within 30 days after each quarter ends. "
    "The Licensee must not sublicense the Software."
)
LEGALESE = (
    "Notwithstanding any provision hereof to the contrary, the obligations of the lessee "
    "hereunder shall constitute independent covenants enforceable irrespective of any "
    "counterclaim, recoupment, defence or other right which the lessee may possess."
)


def test_fkgl_band_rewards_plain_text_over_legalese() -> None:
    band = SimplicityConfig()
    assert fkgl_band_score(PLAIN, band) > fkgl_band_score(LEGALESE, band)
    assert fkgl_band_score(LEGALESE, band) == 0.0


def test_fkgl_band_does_not_reward_choppy_text_below_the_band() -> None:
    # The cheapest low grade is a string of two-word sentences; the lower edge stops that paying.
    band = SimplicityConfig(fkgl_low=5, fkgl_high=9, fkgl_margin=4)
    choppy = "Pay rent. Pay now. Do it. End it."
    assert fkgl(choppy) < band.fkgl_low
    assert fkgl_band_score(choppy, band) < fkgl_band_score(PLAIN, band)


def test_fkgl_band_gives_empty_output_nothing() -> None:
    # Regression: textstat reports FKGL 0 for "", which the old bonus rewarded.
    assert fkgl_band_score("", SimplicityConfig()) == 0.0
    assert fkgl_band_score("   \n", SimplicityConfig()) == 0.0


def test_simplicity_methods() -> None:
    clause, ref = LEGALESE, PLAIN
    s = sentence_sari(clause, PLAIN, [ref])
    b = fkgl_band_score(PLAIN, SimplicityConfig())
    assert simplicity_score(clause, PLAIN, ref, SimplicityConfig(method="sari")) == s
    assert simplicity_score(clause, PLAIN, ref, SimplicityConfig(method="fkgl_band")) == b
    assert simplicity_score(clause, PLAIN, ref, SimplicityConfig(method="both")) == pytest.approx(
        (s + b) / 2
    )
    with pytest.raises(ValueError):
        SimplicityConfig(fkgl_low=9, fkgl_high=5)


# --- entities and format -----------------------------------------------------------


def test_entity_preservation_is_the_normalised_recall() -> None:
    entities = ["$1,000,000", "Acme Inc.", "thirty (30) days", "Territory"]
    assert (
        entity_preservation_score(
            entities, "Acme Inc. pays one million dollars within 30 days in the Territory."
        )
        == 1.0
    )
    assert entity_preservation_score(entities, "Acme Inc. pays $1 million.") == pytest.approx(0.5)
    assert entity_preservation_score([], "anything") == 1.0


CLAUSE = (
    "The Licensee shall pay the Licensor a royalty of 5% of Net Sales within thirty (30) days "
    "after the end of each quarter and shall not sublicense the Software."
)
GOOD = (
    "The Licensee must pay the Licensor 5% of Net Sales within 30 days after each quarter ends. "
    "The Licensee must not sublicense the Software."
)
ENTITIES = ["the Licensee", "the Licensor", "5%", "Net Sales", "thirty (30) days", "the Software"]


def test_format_is_zero_for_a_clean_rewrite() -> None:
    assert format_score(CLAUSE, GOOD, ENTITIES, FormatConfig()) == (0.0, [])


@pytest.mark.parametrize(
    ("output", "problem"),
    [
        ("", "empty"),
        (CLAUSE, "verbatim_copy"),
        (GOOD + " " + " ".join([GOOD] * 2), "runaway_length"),
        ("The Licensee pays. It is quarterly.", "truncation"),
        ("the Licensee the Licensor 5% Net Sales 30 days the Software.", "word_salad"),
        (GOOD.replace("must not sublicense", "may sublicense"), "modality_flip"),
    ],
)
def test_format_penalises_each_problem(output: str, problem: str) -> None:
    value, problems = format_score(CLAUSE, output, ENTITIES, FormatConfig())
    assert problem in problems
    assert -1.0 <= value < 0.0


def test_format_penalties_are_configurable_and_clipped() -> None:
    lenient = FormatConfig(modality_flip_penalty=0.0)
    flipped = GOOD.replace("must not sublicense", "may sublicense")
    assert format_score(CLAUSE, flipped, ENTITIES, lenient)[0] == 0.0
    harsh = FormatConfig(copy_penalty=5.0)
    assert format_score(CLAUSE, CLAUSE, ENTITIES, harsh)[0] == -1.0


# --- the suite -------------------------------------------------------------------


class FakeNLI:
    """Entailment = share of output words found in the clause; records batch sizes."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def entailment(self, premises: Sequence[str], hypotheses: Sequence[str]) -> list[float]:
        self.calls.append(len(premises))
        out = []
        for p, h in zip(premises, hypotheses, strict=True):
            words = set(tokenize(h))
            out.append(len(words & set(tokenize(p))) / len(words) if words else 0.0)
        return out


def test_suite_scores_a_good_rewrite_above_hacks() -> None:
    nli = FakeNLI()
    suite = RewardSuite(RewardConfig(), entailment=nli)
    outputs = [GOOD, CLAUSE, "", "The Licensee pays. It is quarterly."]
    n = len(outputs)
    results = suite.score([CLAUSE] * n, outputs, [GOOD] * n, [ENTITIES] * n)
    good, copy, empty, truncated = results
    assert good.total > max(copy.total, empty.total, truncated.total)
    assert empty.meaning == 0.0 and empty.format == -1.0
    assert "verbatim_copy" in copy.problems
    assert nli.calls == [n - 1]  # one batch, empty output skipped
    assert set(good.to_dict()) == {*REWARD_NAMES, "total", "problems"}


def test_suite_weights_and_zero_weight_ablation() -> None:
    weights = RewardWeights(meaning=0, simplicity=0, entity_preservation=0, format=1)
    suite = RewardSuite(RewardConfig(weights=weights), entailment=FakeNLI())
    (r,) = suite.score([CLAUSE], [GOOD], [GOOD], [ENTITIES])
    assert r.total == r.format == 0.0
    off = RewardSuite(
        RewardConfig(weights=RewardWeights(entity_preservation=0)), entailment=FakeNLI()
    )
    (with_entities,) = RewardSuite(RewardConfig(), entailment=FakeNLI()).score(
        [CLAUSE], [GOOD], [GOOD], [ENTITIES]
    )
    (without,) = off.score([CLAUSE], [GOOD], [GOOD], [ENTITIES])
    assert with_entities.total - without.total == pytest.approx(with_entities.entity_preservation)
    assert without.entity_preservation == with_entities.entity_preservation  # still computed


def test_meaning_can_use_bertscore_against_the_reference() -> None:
    scorer = FakeScorer()
    suite = RewardSuite(
        RewardConfig.model_validate({"meaning": {"method": "bertscore"}}), similarity=scorer
    )
    assert suite.meaning([CLAUSE, CLAUSE], [GOOD, ""], [GOOD, GOOD]) == [pytest.approx(1.0), 0.0]
    assert scorer.calls == [1]


def test_suite_rejects_mismatched_batches() -> None:
    with pytest.raises(ValueError):
        RewardSuite(RewardConfig(), entailment=FakeNLI()).score(
            [CLAUSE], [GOOD, GOOD], [GOOD], [ENTITIES]
        )


def test_grpo_reward_functions_are_named_for_logging_and_read_dataset_columns() -> None:
    from trl.trainer.utils import get_callable_name

    suite = RewardSuite(RewardConfig(), entailment=FakeNLI())
    functions = suite.grpo_reward_functions()
    assert [get_callable_name(f) for f in functions] == list(REWARD_NAMES)
    completions = [[{"role": "assistant", "content": GOOD}], [{"role": "assistant", "content": ""}]]
    columns = {
        "clause": [CLAUSE] * 2,
        "reference": [GOOD] * 2,
        "entities": [ENTITIES] * 2,
        "category": ["x"] * 2,
    }
    expected = suite.score([CLAUSE] * 2, [GOOD, ""], [GOOD] * 2, [ENTITIES] * 2)
    for name, fn in zip(REWARD_NAMES, functions, strict=True):
        values = fn(prompts=[[]] * 2, completions=completions, trainer_state=None, **columns)
        assert values == [getattr(e, name) for e in expected]


def test_completion_text_handles_messages_and_think_blocks() -> None:
    assert completion_text("  a\n b ") == "a b"
    assert (
        completion_text(
            [{"role": "assistant", "content": "<think>\nhmm\n</think>\n\nYou must pay."}]
        )
        == "You must pay."
    )
