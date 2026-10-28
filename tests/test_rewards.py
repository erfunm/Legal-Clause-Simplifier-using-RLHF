from __future__ import annotations

from collections.abc import Sequence

import pytest

from legal_simplifier.config import RewardConfig, RewardWeights
from legal_simplifier.rewards import components
from legal_simplifier.rewards.components import (
    bertscore,
    combine,
    composite_rewards,
    entity_penalty,
    readability_bonus,
    sari,
)
from legal_simplifier.rewards.sari import sentence_sari, tokenize


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


# --- readability ------------------------------------------------------------


def test_readability_rewards_plain_text_over_legalese() -> None:
    plain = "You must pay the rent on time. If you do not, we can end the lease."
    legalese = (
        "Notwithstanding any provision hereof to the contrary, the obligations of the lessee "
        "hereunder shall constitute independent covenants enforceable irrespective of any "
        "counterclaim, recoupment, defence or other right which the lessee may possess."
    )
    assert readability_bonus(plain) > readability_bonus(legalese)
    assert readability_bonus(legalese) == 0.0


def test_readability_gives_empty_output_no_bonus() -> None:
    # Regression: textstat reports FKGL 0 for "", which used to earn the maximum bonus.
    assert readability_bonus("") == 0.0
    assert readability_bonus("   \n") == 0.0


def test_readability_is_capped_at_one() -> None:
    assert readability_bonus("Go. Run. Eat.") <= 1.0


# --- entities -----------------------------------------------------------------


def test_entity_penalty_scales_with_fraction_missing() -> None:
    entities = ["$100", "Acme Inc.", "30 days", "Territory"]
    assert entity_penalty(entities, "Acme Inc. pays $100 within 30 days in the Territory.") == 0.0
    assert entity_penalty(entities, "Acme Inc. pays $100.") == pytest.approx(0.5)
    assert entity_penalty(entities, "", missing_penalty=2.0) == pytest.approx(2.0)


def test_entity_penalty_is_zero_without_entities() -> None:
    assert entity_penalty([], "anything") == 0.0


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


# --- composite -----------------------------------------------------------------


def test_combine_uses_config_weights_and_clips() -> None:
    config = RewardConfig(
        weights=RewardWeights(bertscore=1, sari=1, readability=1, entity_penalty=2)
    )
    value = combine(bertscore=0.5, sari=0.25, readability=0.25, entity_penalty=0.1, config=config)
    assert value == pytest.approx(0.8)
    assert (
        combine(bertscore=1, sari=1, readability=1, entity_penalty=0, config=config)
        == config.clip_max
    )
    assert (
        combine(bertscore=0, sari=0, readability=0, entity_penalty=5, config=config)
        == config.clip_min
    )


def test_reward_config_rejects_inverted_clip() -> None:
    with pytest.raises(ValueError):
        RewardConfig(clip_min=1.0, clip_max=0.0)


def test_composite_rewards_breakdown_and_ordering() -> None:
    clause = "The Licensee shall remit $5,000 to Acme Inc. within thirty (30) days of the Effective Date."
    entities = ["Licensee", "$5,000", "Acme Inc.", "thirty (30) days", "Effective Date"]
    reference = "The Licensee must pay Acme Inc. $5,000 within 30 days of the Effective Date."
    predictions = [reference, "The Licensee must pay some money.", ""]
    results = composite_rewards(
        clauses=[clause] * 3,
        predictions=predictions,
        references=[reference] * 3,
        entities=[entities] * 3,
        config=RewardConfig(),
        scorer=FakeScorer(),
    )
    good, lossy, empty = results
    assert good.entity_penalty == 0.0
    assert lossy.entity_penalty > 0.0
    assert empty.readability == 0.0
    assert good.total > lossy.total > empty.total
    assert set(good.to_dict()) == {"bertscore", "sari", "readability", "entity_penalty", "total"}


def test_composite_rewards_rejects_mismatched_batches() -> None:
    with pytest.raises(ValueError):
        composite_rewards(
            clauses=["a"],
            predictions=["a", "b"],
            references=["a"],
            entities=[[]],
            config=RewardConfig(),
            scorer=FakeScorer(),
        )
